## Dutch Tutor (LoRA)

This project includes a LoRA fine-tuned version of GEITje-7B,
specialized as a Dutch language tutor.

The model:
- Corrects Dutch sentences
- Explains mistakes in simple English
- Provides short Dutch examples

Training was done using:
- Dutch Ultrachat (instruction-style data)
- Simplified Dutch Wikipedia texts (A2 level)

## What it does
Given a Dutch sentence, it returns:
- a corrected version
- a short explanation in simple English
- two correct Dutch example sentences

## Run (Colab / GPU recommended)
```bash
python inference/run_tutor.py --sentence "Ik heb gisteren naar winkel gaan."

