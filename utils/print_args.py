def print_args(args):
    for key, value in sorted(vars(args).items()):
        print(f"{key}: {value}")
