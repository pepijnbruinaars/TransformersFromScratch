from transformer import Transformer


def main() -> None:
    # From the paper
    d_model = 512
    transformer = Transformer(d_model)
    print(transformer)


if __name__ == "__main__":
    main()
