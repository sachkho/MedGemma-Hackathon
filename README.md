# MedGemma × BBBC021 — Hackathon Project (Solve for Healthcare & Life Sciences with Gemma, July 2025)

This repository contains the code and assets used during the Google GDG Paris hackathon
**“Solve for Healthcare & Life Sciences with Gemma” (July 2025)**.  
Goal: fine‑tune a **multimodal** Gemma/MedGemma model on **BBBC021** cell‑painting images to generate concise, structured
captions describing compound, SMILES, concentration, and **mechanism of action (MoA)**.

> ⚠️ Disclaimer: This project is strictly for research & education. **Not for clinical or diagnostic use.**

## Repository structure
```
medgemma-bbbc021-hackathon/
├─ src/
│  ├─ train_medgemma.py    # fine-tuning with LoRA
│  ├─ inference.py         # simple inference helper
│  ├─ eval.py              # automatic metrics (ROUGE/BLEU/BERTScore)
│  └─ ui.py                # Gradio demo app
├─ notebooks/
│  └─ dataset.ipynb        # build JSONL dataset from BBBC021 CSVs
├─ data/
│  ├─ BBBC021_v1_image.csv
│  ├─ BBBC021_v1_compound.csv
│  ├─ BBBC021_v1_moa.csv
│  └─ bbbc021_week1_training.jsonl  # manifest: {"image": <relative_path>, "text": <caption>}
├─ scripts/
│  ├─ train.sh
│  ├─ eval.sh
│  └─ ui.sh
├─ config.yaml
├─ requirements.txt
├─ MODEL_CARD.md
├─ LICENSE (MIT)
└─ .gitignore
```

## Setup
- **Python** 3.10 recommended.
- **CUDA** (optional but recommended for training/inference speed).

```bash
python -m venv .venv && source .venv/bin/activate  # (Linux/Mac)
python -m pip install -U pip
pip install -r requirements.txt
```

## Data
This repo includes only lightweight metadata CSVs and a **JSONL** manifest created from BBBC021.  
You must provide the actual microscopy images locally. The JSONL expects **relative paths** in the `image` field, e.g.:
```
data/Week1_22123/A01_s1_w1.png
```
Place your images under `data/Week1_22123/` (or change paths to match your layout).  
You can regenerate the JSONL using `notebooks/dataset.ipynb`.

## Training
- Configure training in `config.yaml`. Example fields: base model id (e.g., `google/medgemma-4b-it`),
  dataset path, output directory, LoRA settings, learning rate, epochs, batch size.
- Run:
```bash
bash scripts/train.sh
# or
python src/train_medgemma.py --config_file_path config.yaml
```

> Tip: If your script currently loads a remote dataset (e.g., `load_dataset("sachkho/gemma_data")`), switch to your local JSONL:
```python
from datasets import load_dataset
raw_dataset = load_dataset("json", data_files={"train": "data/bbbc021_week1_training.jsonl"}, split="train")
```

## Inference (CLI helper)
```bash
python src/inference.py
```
Use the exposed `infer(image, text)` function inside a Python shell or wire it to your own script.

## Gradio Demo
```bash
bash scripts/ui.sh
# then open the local URL printed by Gradio
```

## Evaluation
```bash
bash scripts/eval.sh
```
This runs ROUGE/BLEU/BERTScore on a sample from your dataset comparing base vs fine‑tuned model.

## Results (sample)
- We observed gains in ROUGE-L and BERTScore vs base MedGemma on Week 1 samples.
- Add your exact numbers here once you run `src/eval.py` on your environment.

## Citing Data
- **BBBC021**: Broad Bioimage Benchmark Collection. Please refer to BBBC’s license and citation guidelines.

## Acknowledgments
- Google Gemma/MedGemma team and GDG Paris organizers.
- Broad Institute’s BBBC for dataset resources.
