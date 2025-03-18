'''Module for loading the car dataset.
'''
from pathlib import Path

import pandas as pd

_TRAIN_FILENAME = 'train.csv'
_TEST_FILENAME = 'test.csv'


def _save_dataset(
    dataset: pd.DataFrame,
    filepath: Path,
) -> None:
    dataset.to_csv(filepath, index=False)


def _load_dataset(filepath: Path,) -> pd.DataFrame:
    return pd.read_csv(filepath)


def save_datasets(
    train_dataset: pd.DataFrame,
    test_dataset: pd.DataFrame,
    basepath: Path,
) -> None:
    '''Saves the train and test datasets to the data_dir.
    '''
    train_filepath = basepath / _TRAIN_FILENAME
    _save_dataset(train_dataset, train_filepath)

    test_filepath = basepath / _TEST_FILENAME
    _save_dataset(test_dataset, test_filepath)


def load_datasets(basepath: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''Loads the train and test datasets from the data_dir.
    '''
    train_filepath = basepath / _TRAIN_FILENAME
    train_dataset = _load_dataset(train_filepath)

    test_filepath = basepath / _TEST_FILENAME
    test_dataset = _load_dataset(test_filepath)

    return train_dataset, test_dataset
