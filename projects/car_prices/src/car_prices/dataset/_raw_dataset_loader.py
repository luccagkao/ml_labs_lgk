'''Module for loading the car dataset.
'''
import zipfile
from pathlib import Path

import pandas as pd
import requests

from ._base import PROJECT_NAME

_DATASET_URL = 'https://www.kaggle.com/api/v1/datasets/download/asinow/car-price-dataset'
_TIMEOUT = 10

_BASENAME = PROJECT_NAME + '_dataset'
_COMPRESSED_FILENAME = _BASENAME + '.zip'
_DATASET_FILENAME = _BASENAME + '.csv'


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
    project_data_dir = data_dir / PROJECT_NAME
    dataset_path = project_data_dir / _DATASET_FILENAME
    if not dataset_path.exists():
        _fetch_and_unpack_car_dataset(
            project_data_dir,
            remove_original=remove_original,
        )
    dataset = pd.read_csv(dataset_path)
    return dataset
