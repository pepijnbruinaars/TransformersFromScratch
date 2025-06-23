from transformer import Transformer


def main() -> None:
    # From the paper
    encoder_blocks = 6
    d_model = 512
    d_ff = 2048
    n_heads = 8
    dropout = 0.1

    transformer = Transformer(
        encoder_blocks=encoder_blocks,
        d_model=d_model,
        d_ff=d_ff,
        n_heads=n_heads,
        dropout=dropout,
    )
    print(transformer)


if __name__ == "__main__":
    main()
