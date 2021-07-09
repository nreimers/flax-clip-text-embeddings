import torch
from transformers import AutoTokenizer
import sys
from PytorchModel import AutoModelForSentenceEmbedding

model = AutoModelForSentenceEmbedding(sys.argv[1])
model.eval()
tokenizer = AutoTokenizer.from_pretrained(sys.argv[1])


### STS Benchmark
import gzip
import csv
sts_dataset_path = 'data/stsbenchmark.tsv.gz'


inp1 = []
inp2 = []
gold_scores = []
with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        inp_example1 = [row['sentence1'], row['sentence2']]

        if row['split'] == 'test':
            inp1.append(row['sentence1'])
            inp2.append(row['sentence2'])
            gold_scores.append(float(row['score']) / 5.0)  # Normalize score to range 0 ... 1

assert len(inp1) == len(gold_scores)

inputs1 = tokenizer(inp1, padding=True, max_length=128, return_tensors="pt")
inputs2 = tokenizer(inp2, padding=True, max_length=128, return_tensors="pt")

with torch.no_grad():
    emb1 = model(**inputs1)
    emb2 = model(**inputs2)
    scores = torch.mm(emb1, emb2.transpose(0, 1))

### Compute cosine + spearman
cosine_scores = []
for i in range(len(scores)):
    cosine_scores.append(scores[i][i])

from scipy.stats import spearmanr
eval_spearman_cosine, _ = spearmanr(gold_scores, cosine_scores)
print("STS test performance: {:.2f}".format(eval_spearman_cosine*100))
