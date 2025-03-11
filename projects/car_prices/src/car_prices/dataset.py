'''Module for loading the car dataset.
'''
import json
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

_DATASET_URL = 'https://www.kaggle.com/api/v1/datasets/download/asinow/car-price-dataset'
_TIMEOUT = 10
_PROJECT_NAME = 'car_price'
_BASENAME = _PROJECT_NAME + '_dataset'
_COMPRESSED_FILENAME = _BASENAME + '.zip'
_DATASET_FILENAME = _BASENAME + '.csv'
_SPLIT_FOLDER = 'processed'
_TRAIN_FILENAME = _BASENAME + '_train.csv'
_TEST_FILENAME = _BASENAME + '_test.csv'
_METADATA_FILENAME = _BASENAME + '_metadata.csv'


def _fetch_car_dataset(raw_dataset_path: Path, project_data_dir: Path) -> None:
    '''Fetches the car dataset from Kaggle and saves it to the data_dir.
    '''
    project_data_dir.mkdir(parents=True, exist_ok=True)
    response = requests.get(_DATASET_URL, timeout=_TIMEOUT)
    response.raise_for_status()
    with open(raw_dataset_path, 'wb') as f:
        f.write(response.content)


def _unpack_car_dataset(raw_dataset_path: Path, project_data_dir: Path) -> None:
    '''Unpacks the car dataset from the data_dir.
    '''
    with zipfile.ZipFile(raw_dataset_path, 'r') as zip_ref:
        zip_ref.extractall(project_data_dir)


def _fetch_and_unpack_car_dataset(
    project_data_dir: Path,
    remove_original: bool,
) -> None:
    '''Fetches and unpacks the car dataset from Kaggle.
    '''
    raw_dataset_path = project_data_dir / _COMPRESSED_FILENAME
    _fetch_car_dataset(raw_dataset_path, project_data_dir)
    _unpack_car_dataset(raw_dataset_path, project_data_dir)
    if remove_original:
        raw_dataset_path.unlink()


def load_car_dataset(
    data_dir: str | Path,
    remove_original: bool = False,
) -> pd.DataFrame:
    '''Loads the car dataset from the data_dir.
    '''
    data_dir = Path(data_dir)
    project_data_dir = data_dir / _PROJECT_NAME
    dataset_path = project_data_dir / _DATASET_FILENAME
    if not dataset_path.exists():
        _fetch_and_unpack_car_dataset(
            project_data_dir,
            remove_original=remove_original,
        )
    dataset = pd.read_csv(dataset_path)
    return dataset


def _get_split_paths(data_dir: str | Path) -> tuple[Path, Path, Path]:
    data_dir = Path(data_dir)
    split_dataset_dir = data_dir / _PROJECT_NAME / _SPLIT_FOLDER
    train_dataset_path = split_dataset_dir / _TRAIN_FILENAME
    test_dataset_path = split_dataset_dir / _TEST_FILENAME
    metadata_path = split_dataset_dir / _METADATA_FILENAME
    return (
        split_dataset_dir,
        train_dataset_path,
        test_dataset_path,
        metadata_path,
    )


def split_train_test_and_save(
    dataset: pd.DataFrame,
    metadata: dict[str, Any],
    data_dir: str | Path,
) -> None:
    '''Splits the dataset into train and test datasets and saves them to the data_dir.
    '''
    train_dataset, test_dataset = train_test_split(
        dataset,
        test_size=metadata['test_size'],
        random_state=metadata['random_state'],
    )

    data_dir = Path(data_dir)

    (
        split_dataset_dir,
        train_dataset_path,
        test_dataset_path,
        metadata_path,
    ) = _get_split_paths(data_dir)

    split_dataset_dir.mkdir(parents=True, exist_ok=True)

    train_dataset.to_csv(train_dataset_path, index=False)
    test_dataset.to_csv(test_dataset_path, index=False)

    with open(metadata_path, 'w', encoding='utf8') as metadata_file:
        json.dump(metadata, metadata_file, indent=4)


_SplitReturns = tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]


def load_car_dataset_split(data_dir: str | Path) -> _SplitReturns:
    '''Loads the train and test datasets and metadata from the data_dir.
    '''
    data_dir = Path(data_dir)

    (
        _,
        train_dataset_path,
        test_dataset_path,
        metadata_path,
    ) = _get_split_paths(data_dir)

    train_dataset = pd.read_csv(train_dataset_path)
    test_dataset = pd.read_csv(test_dataset_path)

    with open(metadata_path, 'r', encoding='utf8') as metadata_file:
        metadata = json.load(metadata_file)

    return train_dataset, test_dataset, metadata
