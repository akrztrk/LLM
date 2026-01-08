# app.py
# Streamlit chat app for a Dutch tutor (base model + LoRA adapter).
# This version enforces a strict output format and validates the "Correct" line.
# It also uses a minimal-edit prompt to prevent paraphrasing.

import os
import re
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel


# -----------------------------
# Configuration
# -----------------------------
@dataclass
class AppConfig:
    base_model_id: str
    adapter_path: str
    device_map: str = "auto"
    max_new_tokens: int = 200
    temperature: float = 0.0
    do_sample: bool = False
    repetition_penalty: float = 1.12
    no_repeat_ngram_size: int = 3


DEFAULT_BASE_MODEL = os.environ.get("BramVanroy/GEITje-7B-ultra", "").strip()
DEFAULT_ADAPTER_PATH = os.environ.get("/content/drive/MyDrive/LLM/models/dutch_tutor_lora_v3_final", "").strip()

if not DEFAULT_BASE_MODEL:
    DEFAULT_BASE_MODEL = "BramVanroy/GEITje-7B-ultra"
if not DEFAULT_ADAPTER_PATH:
    DEFAULT_ADAPTER_PATH = "/content/drive/MyDrive/LLM/models/dutch_tutor_lora_v3_final"

CFG = AppConfig(
    base_model_id=DEFAULT_BASE_MODEL,
    adapter_path=DEFAULT_ADAPTER_PATH,
)

SYSTEM_PROMPT = (
    "You are a Dutch language tutor.\n"
    "Reply in simple English.\n\n"
    "Task:\n"
    "- Fix grammar and word order only.\n"
    "- Do NOT rewrite or paraphrase.\n"
    "- Do NOT shorten the sentence.\n"
    "- Keep the same meaning.\n"
    "- Change as little as possible.\n\n"
    "Output format (always include all sections):\n"
    "- Correct: <one full corrected Dutch sentence>\n"
    "- Explanation: <one short English explanation>\n"
    "- Examples:\n"
    "  - <short Dutch example 1>\n"
    "  - <short Dutch example 2>\n\n"
    "Do not add anything else.\n"
)


# -----------------------------
# Prompt helpers
# -----------------------------
def build_prompt(student_sentence: str) -> str:
    s = (student_sentence or "").strip()
    return (
        f"{SYSTEM_PROMPT}\n"
        f"Student sentence: {s}\n"
        f"Answer:\n"
    )


def extract_after_answer(decoded: str) -> str:
    if not decoded:
        return ""
    if "Answer:" in decoded:
        return decoded.split("Answer:", 1)[-1].strip()
    return decoded.strip()


# -----------------------------
# Parsing helpers
# -----------------------------
def parse_sections(text: str) -> Tuple[Optional[str], Optional[str], List[str]]:
    """
    Extract sections from model output:
      - Correct:
      - Explanation:
      - Examples: (2 bullet lines)
    Returns: (correct, explanation, examples_list)
    """
    if not text:
        return None, None, []

    # Normalize bullets a bit
    t = text.replace("\r\n", "\n")

    correct = None
    explanation = None
    examples: List[str] = []

    # Capture "Correct:" line
    m_c = re.search(r"(?im)^\s*-\s*Correct:\s*(.+?)\s*$", t)
    if m_c:
        correct = m_c.group(1).strip()

    # Capture "Explanation:" line
    m_e = re.search(r"(?im)^\s*-\s*Explanation:\s*(.+?)\s*$", t)
    if m_e:
        explanation = m_e.group(1).strip()

    # Capture examples block lines (bullets)
    # Accept "  - ..." or "- ..." under Examples
    ex_block = re.split(r"(?im)^\s*-\s*Examples:\s*$", t)
    if len(ex_block) >= 2:
        tail = ex_block[1]
        for line in tail.splitlines():
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^(?:[-•]\s+|\d+\.\s+)(.+)$", line)
            if m:
                ex = m.group(1).strip()
                # Stop if we hit another section accidentally
                if ex.lower().startswith(("correct:", "explanation:", "examples:")):
                    continue
                examples.append(ex)
            if len(examples) >= 2:
                break

    return correct, explanation, examples


# -----------------------------
# Validation / fallback helpers
# -----------------------------
def looks_like_full_sentence(nl: str) -> bool:
    """
    Lightweight sanity check to avoid fragments like 'ik vaak televisie.'
    """
    if not nl:
        return False
    s = nl.strip()
    if len(s.split()) < 4:
        return False
    if not s.endswith((".", "!", "?")):
        return False
    # Must contain a verb-ish token for simple check (very heuristic)
    # This is intentionally simple; we only want to catch obvious fragments.
    verb_hints = ["ben", "is", "zijn", "heb", "heeft", "gaan", "ga", "gaat", "kijk", "kijkt", "doe", "doet", "lees", "leest"]
    if not any(v in s.lower().split() for v in verb_hints):
        return False
    return True


def minimal_word_order_fix(student: str) -> str:
    """
    Simple rule-based fix for a common Dutch mistake:
    'In het weekend ik kijk ...' -> 'In het weekend kijk ik ...'
    This is a fallback if the model returns a bad 'Correct' line.
    """
    s = (student or "").strip()

    # Pattern: (prefix) + "ik" + (verb) + rest
    # Example: "In het weekend ik kijk vaak tv."
    m = re.match(r"^(In het weekend)\s+ik\s+(\w+)\s+(.*)$", s, flags=re.IGNORECASE)
    if m:
        prefix = m.group(1)
        verb = m.group(2)
        rest = m.group(3).strip()
        # Keep original casing of prefix as typed
        fixed = f"{prefix} {verb} ik {rest}"
        # Ensure sentence ends with a period if it had one
        if not fixed.endswith((".", "!", "?")) and s.endswith((".", "!", "?")):
            fixed += "."
        return fixed

    return s


def format_output(correct: str, explanation: str, examples: List[str]) -> str:
    ex1 = examples[0] if len(examples) > 0 else "(no example)"
    ex2 = examples[1] if len(examples) > 1 else "(no example)"
    return (
        f"- Correct: {correct}\n"
        f"- Explanation: {explanation}\n"
        f"- Examples:\n"
        f"  - {ex1}\n"
        f"  - {ex2}"
    )


def force_format(student_sentence: str, raw_model_text: str) -> str:
    """
    Build a stable UI response:
    - Parse model output into sections.
    - Validate 'Correct' (avoid fragments / paraphrases).
    - Fallback to minimal rule-based correction if needed.
    """
    s = (student_sentence or "").strip()
    correct, explanation, examples = parse_sections(raw_model_text)

    # Defaults if missing
    if not explanation:
        explanation = "I corrected the grammar and word order with minimal changes."
    if len(examples) < 2:
        # Provide safe defaults; can be improved later
        examples = (examples + ["In het weekend kijk ik vaak tv.", "Morgen ga ik naar de supermarkt."])[:2]

    # Validate 'Correct'
    if not correct or not looks_like_full_sentence(correct):
        # Fallback to minimal rule-based fix
        correct = minimal_word_order_fix(s)
        # If still not a sentence, keep original
        if not looks_like_full_sentence(correct):
            correct = s if s.endswith((".", "!", "?")) else (s + ".")

    # If the model "correct" is too different (very rough check), keep minimal fix
    # This prevents aggressive paraphrasing.
    # Heuristic: require at least 50% word overlap with the original.
    orig_words = set(re.findall(r"\w+", s.lower()))
    corr_words = set(re.findall(r"\w+", correct.lower()))
    if orig_words and (len(orig_words & corr_words) / max(1, len(orig_words))) < 0.5:
        correct = minimal_word_order_fix(s)

    return format_output(correct, explanation, examples)


# -----------------------------
# Model loading & generation
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer(base_model_id: str, adapter_path: str, device_map: str):
    if base_model_id.startswith("PUT_") or adapter_path.startswith("PUT_"):
        raise ValueError(
            "BASE_MODEL_ID / ADAPTER_PATH are not set. "
            "Set them in app.py or via environment variables."
        )

    tok = AutoTokenizer.from_pretrained(base_model_id)

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        device_map=device_map,
    )

    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model, tok


def generate_raw(model, tokenizer, student_sentence: str, cfg: AppConfig) -> str:
    prompt = build_prompt(student_sentence)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_cfg = GenerationConfig(
        max_new_tokens=cfg.max_new_tokens,
        do_sample=cfg.do_sample,
        temperature=cfg.temperature,
        repetition_penalty=cfg.repetition_penalty,
        no_repeat_ngram_size=cfg.no_repeat_ngram_size,
        pad_token_id=tokenizer.eos_token_id,
    )

    with torch.no_grad():
        out = model.generate(**inputs, generation_config=gen_cfg)

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    return extract_after_answer(decoded)


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Dutch Tutor Chat", page_icon="🧡", layout="centered")

st.title("Dutch Tutor Chat")
st.caption("Minimal grammar correction + brief explanation + 2 examples (format enforced).")

with st.sidebar:
    st.header("Settings")
    st.text_input("Base model", value=CFG.base_model_id, key="base_model_id")
    st.text_input("Adapter path", value=CFG.adapter_path, key="adapter_path")
    st.slider("Max new tokens", 80, 400, CFG.max_new_tokens, 10, key="max_new_tokens")
    st.checkbox("Sampling (do_sample)", value=CFG.do_sample, key="do_sample")
    st.slider("Temperature", 0.0, 1.5, float(CFG.temperature), 0.05, key="temperature")
    st.slider("Repetition penalty", 1.0, 1.6, float(CFG.repetition_penalty), 0.01, key="repetition_penalty")
    st.slider("No-repeat ngram size", 0, 6, int(CFG.no_repeat_ngram_size), 1, key="no_repeat_ngram_size")

    st.divider()
    st.write("Runtime")
    st.write(f"CUDA available: `{torch.cuda.is_available()}`")
    if torch.cuda.is_available():
        st.write(f"GPU: `{torch.cuda.get_device_name(0)}`")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_text = st.chat_input("Write a Dutch sentence...")

if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    with st.chat_message("assistant"):
        t0 = time.time()

        cfg_runtime = AppConfig(
            base_model_id=st.session_state.base_model_id.strip(),
            adapter_path=st.session_state.adapter_path.strip(),
            device_map=CFG.device_map,
            max_new_tokens=int(st.session_state.max_new_tokens),
            do_sample=bool(st.session_state.do_sample),
            temperature=float(st.session_state.temperature),
            repetition_penalty=float(st.session_state.repetition_penalty),
            no_repeat_ngram_size=int(st.session_state.no_repeat_ngram_size),
        )

        raw = ""
        try:
            model, tok = load_model_and_tokenizer(
                cfg_runtime.base_model_id,
                cfg_runtime.adapter_path,
                cfg_runtime.device_map,
            )
            raw = generate_raw(model, tok, user_text, cfg_runtime)
        except Exception as e:
            st.error(f"Model error: {e}")

        final = force_format(user_text, raw)
        st.markdown(final)

        dt = time.time() - t0
        st.caption(f"Response time: {dt:.2f}s")

    st.session_state.messages.append({"role": "assistant", "content": final})
