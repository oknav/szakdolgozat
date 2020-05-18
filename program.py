import argparse
from config import *
from autoencoder import *
from get_info import *
from validation import *
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--autoencoder', nargs=2, metavar=('train', 'test'),
                        help='Train or Test on given models and images give in config.py.',
                        required=False)
    parser.add_argument('-r', '--results', action='store_true',
                        help='Get results of training.',
                        required=False)
    parser.add_argument('-v', '--validate', metavar=('classify'),
                        help='Validate results. Set Classify True for validate with classification.',
                        required=False)
    args = parser.parse_args()

    if args.autoencoder:
        conf = Config()
        for i in range(NUM_OF_TRAININGS):
            conf.TRAIN_NUMBER = i
            autoencoder(args.autoencoder[0], args.autoencoder[1])
            if args.results:
                analyze(conf.TRAIN_NUMBER)
            if i < NUM_OF_TRAININGS-1:
                for file in os.listdir(MODELS_PATH):
                    file_path = os.path.join(MODELS_PATH, file)
                    try:
                        os.remove(file_path)
                    except:
                        print('Could not remove %s.' % file_path)
            else:
                pass

    if args.results and not args.autoencoder:
        analyze(conf.TRAIN_NUMBER)

    if args.validate:
        validate(args.validate)

