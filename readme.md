## Resuming training

You can resume training from an existing run using the helper script `src/translator/resume.py`.

Example:

```bash
python -m src.translator.resume \
	--run-folder models/transformer/20260113_161242 \
	--checkpoint transformer_epoch_7.pt \
	--additional-epochs 5 \
	--device cuda
```

If `--run-folder` is omitted the script uses the most recent run folder. If `--checkpoint` is omitted it will pick the latest checkpoint in the run folder. The script restores model weights, optimizer state (when possible), and RNG states saved in the checkpoint.

# Transformers from scratch

Wanted to do something on my Friday night, so I decided to turn off code suggestions and started working on this from scratch in PyTorch, using the original Attention Is All You Need paper as reference. Also added a lot of type hints to make it easier to read. 3B1B's videos on transformers were also a great help in understanding the matrix multiplications.

Now that I have the architecture down, I will work on a translation task between Dutch and English, because the original paper was also about translation. I will use the en-nl [opus_books](https://huggingface.co/datasets/Helsinki-NLP/opus_books/viewer/en-nl?views%5B%5D=en_nl) dataset from Hugging Face.

Packages used are PyTorch and HuggingFace. To install them and their dependencies, run:
`pip install -r requirements.txt`.
