'''Preprocess the data and save it to the data directory. '''
from pathlib import Path

from lab01.config import DATA_DIR
from lab01.dataloader import load_housing_data, save_preprocessed_data
from lab01.preprocess import preprocess_data


def pipeline(data_dir: Path) -> None:
    '''Preprocess the data and save it to the data directory.'''
    data = load_housing_data(data_dir)
    preprocessed_data = preprocess_data(data)
    save_preprocessed_data(preprocessed_data, data_dir)


# pylint: disable=missing-function-docstring
def main():
    data_dir = DATA_DIR
    pipeline(data_dir)

if __name__ == '__main__':
    main()
