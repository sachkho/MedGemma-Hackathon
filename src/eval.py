import datasets
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

model_id = "google/medgemma-4b-it"

model_base = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)


our_model = AutoModelForImageTextToText.from_pretrained(
    "Eliot04/CellGemma-3B-v2",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

dataset=datasets.load_dataset("sachkho/gemma_data")

random_samples = dataset.shuffle(seed=42).select(range(10))

labels=[]
result_base=[]
result_ours=[]

for sample in random_samples:
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an expert biologist."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "explain this image."},
                {"type": "image", "image": sample["image"]}
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model_base.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model_base.generate(**inputs, max_new_tokens=200, do_sample=False)
        generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)
    result_base.append(decoded)

    with torch.inference_mode():
        generation = our_model.generate(**inputs, max_new_tokens=200, do_sample=False)
        generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)
    result_ours.append(decoded)

    labels.append(sample["text"])


import evaluate

# Load metrics
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")

# Evaluate base model
rouge_base = rouge.compute(predictions=result_base, references=labels)
bleu_base = bleu.compute(predictions=result_base, references=[[ref] for ref in labels])
bertscore_base = bertscore.compute(predictions=result_base, references=labels, lang="en")

# Evaluate our model
rouge_ours = rouge.compute(predictions=result_ours, references=labels)
bleu_ours = bleu.compute(predictions=result_ours, references=[[ref] for ref in labels])
bertscore_ours = bertscore.compute(predictions=result_ours, references=labels, lang="en")

# Print results
print("Base Model:")
print("  ROUGE:", rouge_base)
print("  BLEU:", bleu_base)
print("  BERTScore:", {
    "precision": sum(bertscore_base['precision']) / len(bertscore_base['precision']),
    "recall": sum(bertscore_base['recall']) / len(bertscore_base['recall']),
    "f1": sum(bertscore_base['f1']) / len(bertscore_base['f1']),
})

print("\nOur Model:")
print("  ROUGE:", rouge_ours)
print("  BLEU:", bleu_ours)
print("  BERTScore:", {
    "precision": sum(bertscore_ours['precision']) / len(bertscore_ours['precision']),
    "recall": sum(bertscore_ours['recall']) / len(bertscore_ours['recall']),
    "f1": sum(bertscore_ours['f1']) / len(bertscore_ours['f1']),
})

