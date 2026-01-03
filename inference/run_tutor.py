import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def build_prompt(sentence: str) -> str:
    return (
        "You are a Dutch language tutor.\n"
        "Rules:\n"
        "- Reply in simple English.\n"
        "- Correct the Dutch sentence.\n"
        "- Explain the mistake briefly and correctly.\n"
        "- Give exactly 2 correct Dutch examples.\n"
        "- Do NOT continue the conversation.\n"
        "- Do NOT ask for another sentence.\n\n"
        f"Student sentence: {sentence}\n"
        "Answer:\n"
        "Correct sentence:\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentence", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="BramVanroy/GEITje-7B-ultra")
    parser.add_argument("--adapter", type=str, default="/content/drive/MyDrive/LLM/models/dutch_tutor_lora_v1/adapter")
    parser.add_argument("--max_new_tokens", type=int, default=180)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        load_in_4bit=True,
    )

    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    prompt = build_prompt(args.sentence)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
        )

    print(tokenizer.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
