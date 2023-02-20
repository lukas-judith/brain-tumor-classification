import argparse

from training import TrainingAlgorithm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3)
    parser.add_argument("-m", "--model", type=str, default="CNN")
    parser.add_argument("-dp", "--data_path", type=str, default="./data")
    parser.add_argument("-ts", "--test_size", type=float, default=0.2)
    parser.add_argument(
        "--reload_data",
        type=bool,
        default=False,
        help="Choose to either load train/test data from existing .npy file, \
            or create new .npy files from the brain tumor image dataset.",
    )
    parser.add_argument("-bs", "--batch_size", type=int, default=16)
    parser.add_argument("-n", "--epochs", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    algo = TrainingAlgorithm(args=args)
    algo.train()
