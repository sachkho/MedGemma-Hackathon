# CellGemma — MedGemma × BBBC021  
*Google GDG Paris Hackathon, July 2025*

> Fine-tuning **MedGemma** (Google’s multimodal vision–language model) on **BBBC021 cell-painting images** with LoRA adapters to generate structured biological captions. Built during the Google GDG Paris *Solve for Healthcare & Life Sciences* Hackathon 2025.

---

## Motivation

Drug discovery today is **slow and costly**:
- It takes **10+ years** on average for a new therapy to reach patients.  
- Scientists must test **millions of compounds** to answer basic questions like *“Does it bind? Does it kill cancer cells?”*  
- High-throughput screening produces **hundreds of terabytes** of microscopy data.  

Yet researchers cannot easily “see” the **full cellular response** to each compound. This creates a bottleneck in identifying promising candidates and delays innovation.  

---

## Our Approach

**Goal**: help biologists interpret microscopy data more efficiently by generating **structured, interpretable captions** from cell images.  

- **Dataset**: BBBC021 (Broad Bioimage Benchmark Collection), which captures morphological changes in cells under drug treatment.  
- **Model**: Fine-tuned Google **MedGemma** (base: `google/medgemma-4b-it`) with **LoRA adapters** for efficient training.  
- **Task**: For each cell-painting image, generate captions including:
  - **Compound name**  
  - **SMILES representation**  
  - **Concentration**  
  - **Mechanism of Action (MoA)**  

---

## Why It Matters

- ⚡ **Accelerates research**: enables rapid screening of compound libraries.  
- 🔍 **Improves interpretability**: captions bridge the gap between raw microscopy images and biological insights.  
- 💊 **Supports applications**: drug repurposing, toxicity prediction, and precision medicine.  

By combining **AI + biology**, CellGemma demonstrates how multimodal models can make **massive biomedical datasets** more actionable.  

---

## Technical Architecture

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

```
BBBC021 images + metadata --> JSONL manifest
            │
            v
Fine-tune MedGemma with LoRA adapters
            │
            ├── Evaluation (ROUGE, BLEU, BERTScore)
            └── Gradio Demo UI (qualitative testing)
```

---

## Repository Structure

```
finetune/       # LoRA training code + config.yaml
demo_ui/        # Gradio demo + inference helper
evaluation/     # Metrics scripts (ROUGE, BLEU, BERTScore)
notebooks/      # Dataset builder notebook
data/           # CSV metadata + JSONL manifest (no raw images)
docs/           # Pitch outline + architecture diagram
requirements.txt
LICENSE
README.md
```

Each subfolder contains its own README.

---

## Getting Started

### Windows (PowerShell)
```powershell
# 1) Create & activate a venv
python -m venv .venv
.venv\Scripts\activate.bat

# 2) Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3) Place microscopy images
# Put raw BBBC021 images under data\Week1_22123\
# (or update the JSONL manifest paths)

# 4) Fine-tune
python finetune\train_medgemma.py --config_file_path finetune\config.yaml

# 5) Launch Gradio demo
python demo_ui\ui.py

# 6) Evaluate results
python evaluation\eval.py
```

### Linux / macOS
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python finetune/train_medgemma.py --config_file_path finetune/config.yaml
python demo_ui/ui.py
python evaluation/eval.py
```

---

## Results (Hackathon Prototype)

- Achieved higher **ROUGE-L** and **BERTScore** vs base MedGemma on BBBC021 Week 1 samples.  
- Demo UI enabled qualitative evaluation by scientists (image → caption).  

*(Full metrics in [`evaluation/`](evaluation/)).*

---

## Data

This repository includes only:
- CSV metadata (`compound.csv`, `image.csv`, `moa.csv`)  
- JSONL manifest (`bbbc021_week1_training.jsonl`)  

⚠️ The raw microscopy images from **BBBC021** are **not redistributed**. Download them from the Broad Bioimage Benchmark Collection and place them under `data/Week1_22123/`.

---

## License

- Code: MIT License (see [LICENSE](LICENSE))  
- Dataset: BBBC021 follows its own license and citation rules.  

---

## Acknowledgments

- Google DeepMind / Gemma team  
- GDG Paris organizers of *Solve for Healthcare & Life Sciences*  
- Broad Institute (BBBC) for dataset resources  
