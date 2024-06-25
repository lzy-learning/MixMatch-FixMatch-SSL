import time

from MixMatch.train import train as mixmatch_train
from MixMatch.test import test as mixmatch_test
from FixMatch.train import train as fixmatch_train
from FixMatch.test import test as fixmatch_test
from MixMatchConfig import config as mixmatch_config
from FixMatchConfig import config as fixmatch_config
import argparse


def main():
    # Create the ArgumentParser object
    parser = argparse.ArgumentParser(description='Parse command line arguments for the project main function')

    # Add an argument for the algorithm selection
    parser.add_argument('--algorithm', type=str, choices=['mix', 'fix'],
                        default='mix', help='Specify the algorithm to use (MixMatch or FixMatch)')

    # Add an argument for the amount of labeled data, with choices restricted to [40, 250, 4000]
    parser.add_argument('--labeled_num', type=int, choices=[40, 250, 4000],
                        default=250, help='Specify the amount of labeled data, choose from: 40, 250, 4000')

    # Add an argument for specifying whether train or test
    parser.add_argument('--train', type=bool, choices=[True, False], default=True, help='True for training the model.')

    # Parse the command line arguments
    args = parser.parse_args()

    # Print the parsed arguments, replace with actual logic as needed
    print(f"Selected algorithm: {args.algorithm}")
    print(f"Amount of labeled data: {args.labeled_num}")
    if args.algorithm == 'mix':
        config = mixmatch_config
        train = mixmatch_train
        test = mixmatch_test
    else:
        config = fixmatch_config
        train = fixmatch_train
        test = fixmatch_test

    print("train iteration: {}".format(config.total_epoch * config.train_iteration))
    config.labeled_num = args.labeled_num
    if args.train:
        train()
    else:
        test()
    if config.device.type == 'cpu':
        print('Using CPU')
    else:
        print('Using GPU')


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print('consume time: {:.2f}s'.format(end - start))
