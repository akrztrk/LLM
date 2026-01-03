# Dutch Tutor (QLoRA)

Educational project: a Dutch language tutor built by adapting an open-source LLM with QLoRA (LoRA adapters).

## What it does
Given a Dutch sentence, it returns:
- a corrected version
- a short explanation in simple English
- two correct Dutch example sentences

## Run (Colab / GPU recommended)
```bash
python inference/run_tutor.py --sentence "Ik heb gisteren naar winkel gaan."

