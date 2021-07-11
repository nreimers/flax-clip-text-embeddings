from modeling_hybrid_clip import FlaxSE
from transformers import AutoTokenizer
import jax.numpy as jnp
import jax
import sys

model = FlaxSE.from_pretrained(sys.argv[1])
tokenizer = AutoTokenizer.from_pretrained(sys.argv[1])
inputs1 = tokenizer(["Tensorflow is a super framework", "A man eats spaghetti", "Berlin is the capital of Germany"], padding="max_length", max_length=128, return_tensors="np")
inputs2 = tokenizer(["Pytorch is the better software", "A woman eats pasta for dinner", "The largest city in Germany is Berlin"], padding="max_length", max_length=128, return_tensors="np")

@jax.jit
def run_model(input_ids1, input_ids2, attention_mask1, attention_mask2):
    return model(
       input_ids1=input_ids1,
       input_ids2=input_ids2,
       attention_mask1=attention_mask1,
       attention_mask2=attention_mask2,
       normalize_embeds=True
    )

out = run_model(
    input_ids1=inputs1['input_ids'],
    input_ids2=inputs2['input_ids'],
    attention_mask1=inputs1["attention_mask"],
    attention_mask2=inputs2["attention_mask"],
)
print(out.text_embeds1.shape)
print(jnp.matmul(out.text_embeds1, out.text_embeds2.T) )

#from transformers import FlaxAutoModel, AutoConfig
#config = AutoConfig.from_pretrained('nreimers/BERT-Tiny_L-2_H-128_A-2')
#model = FlaxAutoModel.from_pretrained('output', config=config)
#print(model)


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

inputs1 = tokenizer(inp1, padding="max_length", max_length=128, return_tensors="np")
inputs2 = tokenizer(inp2, padding="max_length", max_length=128, return_tensors="np")

out = run_model(
    input_ids1=inputs1['input_ids'],
    input_ids2=inputs2['input_ids'],
    attention_mask1=inputs1["attention_mask"],
    attention_mask2=inputs2["attention_mask"],
)

print(out.text_embeds1.shape)

text_embeds1 = out.text_embeds1 / jnp.linalg.norm(out.text_embeds1, axis=-1, keepdims=True)
text_embeds2 = out.text_embeds2 / jnp.linalg.norm(out.text_embeds2, axis=-1, keepdims=True)
scores = jnp.matmul(text_embeds1, text_embeds2.T) * model.params["logit_scale"]

# scores = out.logits_per_text1
cosine_scores = []
for i in range(len(scores)):
    cosine_scores.append(scores[i][i])

from scipy.stats import spearmanr
eval_spearman_cosine, _ = spearmanr(gold_scores, cosine_scores)
print("STS test performance: {:.2f}".format(eval_spearman_cosine*100))
