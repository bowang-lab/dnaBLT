from args import DataloaderArgs


if __name__ == "__main__":
    # main()
    args = DataloaderArgs()
    train_loader = iter(args.build_from_rank(0, 1))
    next(train_loader)
    next(train_loader)
