# CellGemma — MedGemma fine-tuned on BBBC021 (Hackathon)

**Task:** Multimodal image → text (cell-painting images with mechanism-of-action captions)  
**Base model:** `google/medgemma-4b-it`  
**Adapter:** LoRA (q_proj, k_proj, v_proj, o_proj, gate/up/down_proj)

## Data
- Source: BBBC021 (Broad Bioimage Benchmark Collection)
- This repo includes only light metadata CSVs and a JSONL manifest. Raw microscopy images are not redistributed.

## Training
- See `config.yaml` and `scripts/train.sh`.

## Evaluation
- `src/eval.py` computes ROUGE/BLEU/BERTScore on a held-out sample.

## Limitations & Risks
- Hallucinations possible on OOD images.
- Not for diagnostic use.
