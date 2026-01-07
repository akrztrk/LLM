import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

SYSTEM = (
    "You are a Dutch language tutor.\n"
    "Rules:\n"
    "- Reply in simple English.\n"
    "- Correct the Dutch sentence.\n"
    "- Explain the mistake briefly.\n"
    "- Give 2 short Dutch examples.\n"
)

DEFAULT_BASE_MODEL = "BramVanroy/GEITje-7B-ultra"

def build_prompt(student_sentence: str) -> str:
     return f"""
You are a Dutch language tutor.
Reply in simple English.

Example:
Student sentence: Ik heb gisteren naar winkel gaan.
Answer:
- Correct: Ik ben gisteren naar de winkel gegaan.
- Explanation: 'Heb' is not used with movement verbs in the past. Dutch uses 'ben gegaan'.
- Examples:
  - Ik ben naar de supermarkt gegaan.
  - Ik ben gisteren naar huis gegaan.

Now do the same.

Student sentence: {sentence}
Answer:
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    parser.add_argument(
        "--adapter_path",
        default="/content/drive/MyDrive/LLM/models/dutch_tutor_lora_v3_final",
        help="Path to the LoRA adapter folder (contains adapter_config.json).",
    )
    parser.add_argument("--sentence", default="Ik heb gisteren naar winkel gaan.")
    parser.add_argument("--max_new_tokens", type=int, default=180)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    prompt = build_prompt(args.sentence)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    # Print only the assistant-style part (after "Answer:")
    if "Answer:" in text:
        text = text.split("Answer:", 1)[-1].strip()
    print(text)

if __name__ == "__main__":
    main()
