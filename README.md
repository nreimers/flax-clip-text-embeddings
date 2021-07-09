# Hybrid-CLIP as text embedding model
Hybrid-CLIP adapted to a sentence embedding format.

Training data are text pairs: [text1, text2]

Passes both text pairs through the same text encoder.

## Flax Script
Run with:
```
python run_flax_train.py
```

Note: The batch size is 256 per device. Ensure that the model just runs on one device.

## Pytorch Script
Run with:
```
python run_pytorch_train.py
```

## Evaluation

Run:
```
python eval_flax.py output/flax-model
python eval_pytorch.py output/pytorch-model
```

This evaluates the model on  the STS benchmark test dataset.

Performances on STS benchmark test set (Spearman correlation): \
Without training: 9.02 \
With training: 42.30  \
PyTorch reference performance: 75.12