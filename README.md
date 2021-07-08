# Hybrid-CLIP as text embedding model
Hybrid-CLIP adapted to a sentence embedding format.

Training data are text pairs: [text1, text2]

Passes both text pairs through the same text encoder.

Run with:
```
python run_hybrid_clip.py --output_dir output/
```

## Evaluation

Run:
```
python eval.py output/
```

This evaluates the model on  the STS benchmark test dataset.

Performances on STS benchmark test set (Spearman correlation): \
Without training: 9.02 \
With training: 39.84  \
PyTorch reference performance: 77.15