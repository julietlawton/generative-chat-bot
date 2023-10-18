from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader, Dataset, TensorDataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import torch

#Loading the GPT-2 model
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")

#Loading the GPT-2 tokenizer
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

#Example text generation
input_text = "Once upon a time"
input_ids = gpt2_tokenizer.encode(input_text, return_tensors="pt")
output = gpt2_model.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50, top_p=0.95)

for sequence in output:
    text = gpt2_tokenizer.decode(sequence, skip_special_tokens=True)
    print(text)

text = "This is an example sentence."
input_ids = gpt2_tokenizer.encode(text, return_tensors="pt")

with torch.no_grad():
    outputs = gpt2_model(input_ids, labels=input_ids)

#Calculating Perplexity
with torch.no_grad():
    outputs = gpt2_model(input_ids, labels=input_ids)
loss = outputs.loss.item()
print(f"Perplexity: {loss}")

#Calculating BLEU Score
reference = ["This is an example sentence."]
candidate = "This is a sample sentence."

reference = [reference]
candidate = candidate.split()

smoothing = SmoothingFunction().method1
score = sentence_bleu(reference, candidate, smoothing_function=smoothing)
print(f"Smoothed BLEU Score: {score}")

#Calculating ROUGE Score
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

scores = scorer.score(reference, candidate)

print("ROUGE-1 F1 Score:", scores['rouge1'].fmeasure)
print("ROUGE-2 F1 Score:", scores['rouge2'].fmeasure)
print("ROUGE-L F1 Score:", scores['rougeL'].fmeasure)
