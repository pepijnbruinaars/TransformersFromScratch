# Transformers from scratch

Wanted to do something on my Friday night, so I decided to turn off code suggestions and started working on this from scratch in PyTorch, using the original Attention Is All You Need paper as reference. Also added a lot of type hints to make it easier to read. 3B1B's videos on transformers were also a great help in understanding the matrix multiplications.

Now that I have the architecture down, I will work on a translation task between Dutch and English, because the original paper was also about translation. I will use the en-nl [opus_books](https://huggingface.co/datasets/Helsinki-NLP/opus_books/viewer/en-nl?views%5B%5D=en_nl) dataset from Hugging Face.

Packages used are PyTorch and HuggingFace. To install them and their dependencies, run:
`pip install -r requirements.txt`.
