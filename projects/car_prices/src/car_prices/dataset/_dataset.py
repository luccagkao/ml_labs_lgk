'''Module for loading the car dataset.
'''
from pathlib import Path

import pandas as pd

from ._base import PROJECT_NAME, SPLIT_FOLDER
from ._metadata import ExperimentConfig, load_metadata, save_metadata
from ._train_test_datasets import load_datasets, save_datasets
from ._train_test_split import split_train_test


def _get_basepath(data_dir: str | Path) -> Path:
    data_path = Path(data_dir)
    return data_path / PROJECT_NAME / SPLIT_FOLDER


def split_train_test_and_save(
    dataset: pd.DataFrame,
    metadata: ExperimentConfig,
    data_dir: str | Path,
) -> None:
    '''Splits the dataset into train and test sets and saves them to the data_dir.
    '''
    train_dataset, test_dataset = split_train_test(
        dataset=dataset,
        test_size=metadata.test_size,
        random_state=metadata.random_state,
    )
    basepath = _get_basepath(data_dir)
    basepath.mkdir(parents=True, exist_ok=True)
    save_datasets(train_dataset, test_dataset, basepath)
    save_metadata(metadata, basepath)


def load_car_dataset_split(
    data_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, ExperimentConfig]:
    '''Loads the train and test datasets and metadata from the data_dir.
    '''
    basepath = _get_basepath(data_dir)
    train_dataset, test_dataset = load_datasets(basepath)
    metadata = load_metadata(basepath)
    return train_dataset, test_dataset, metadata
