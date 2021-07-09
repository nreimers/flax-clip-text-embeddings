import argparse
import gzip
import json
import logging
import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from PytorchModel import AutoModelForSentenceEmbedding

#### Just some code to print debug information to stdout
class LoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout


from transformers import (
    AdamW,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)

def main():
    # Instantiate the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='nreimers/MiniLM-L6-H384-uncased')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--batch_size_pairs', type=int, default=256)
    parser.add_argument('--batch_size_triplets', type=int, default=256)
    parser.add_argument('data', default='data/AllNLI_2cols.jsonl.gz')
    parser.add_argument('output_dir', default='output/pytorch-model')
    args = parser.parse_args()
    train_function(args)



def train_function(args):
    model = AutoModelForSentenceEmbedding(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    ## Load train data
    dataset = []
    with gzip.open(args.data, 'rt', encoding='utf8') as fIn:
        for line in fIn:
            data = json.loads(line.strip())

            if isinstance(data, dict):
                data = data['texts']

            dataset.append(data)
            if len(dataset) >= (args.steps * args.batch_size_pairs):
                break


    def collate_fn(batch):
        num_texts = len(batch[0])
        texts = [[] for _ in range(num_texts)]
        for example in batch:
            for idx, text in enumerate(example):
                texts[idx].append(text)

        sentence_features = []
        for idx in range(num_texts):
            tokenized = tokenizer(texts[idx], return_tensors="pt", max_length=128, truncation=True, padding=True)
            sentence_features.append(tokenized)

        return sentence_features

    train_dataloader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size_pairs, collate_fn=collate_fn)

    print("len", len(train_dataloader))
    ### Train Loop
    #model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    model.cuda()

    # Instantiate optimizer
    optimizer = AdamW(params=model.parameters(), lr=2e-5, correct_bias=True)

    # Prepare everything
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(train_dataloader),
    )

    # Now we train the model
    cross_entropy_loss = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    max_grad_norm = 1

    model.train()
    for batch in tqdm.tqdm(train_dataloader):

        text1, text2 = batch
        with autocast():
            embeddings_a = model(**text1.to('cuda'))
            embeddings_b = model(**text2.to('cuda'))

            scores = torch.mm(embeddings_a, embeddings_b.transpose(0, 1)) * 20      #We use a constant scale factor
            labels = torch.tensor(range(len(scores)), dtype=torch.long, device=embeddings_a.device)  # Example a[i] should match with b[i]
      
            loss = (cross_entropy_loss(scores, labels) + cross_entropy_loss(scores.transpose(0, 1), labels)) / 2
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad()
        lr_scheduler.step()


    #Save
    print("save model to:", args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    if isinstance(model, torch.nn.DataParallel):
        model.module.save_pretrained(args.output_dir)
    else:
        model.save_pretrained(args.output_dir)




if __name__ == "__main__":
    main()