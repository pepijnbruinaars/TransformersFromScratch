from transformer import Transformer


def main() -> None:
    # Model parameters from the paper
    n_blocks = 6
    d_model = 512
    d_ff = 2048
    n_heads = 8
    dropout = 0.1

    # Dataset dependent parameters
    source_length = 100
    target_length = 100
    source_vocabulary_size = 10000
    target_vocabulary_size = 10000

    transformer = Transformer(
        n_blocks=n_blocks,
        d_model=d_model,
        d_ff=d_ff,
        n_heads=n_heads,
        dropout=dropout,
        source_length=source_length,
        target_length=target_length,
        source_vocabulary_size=source_vocabulary_size,
        target_vocabulary_size=target_vocabulary_size,
    )
    print(transformer)


if __name__ == "__main__":
    main()
