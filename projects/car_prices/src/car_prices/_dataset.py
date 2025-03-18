'''Module for loading the car dataset.
'''
import json
import shutil
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

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
_TRAIN_FILENAME = 'train.csv'
_TEST_FILENAME = 'test.csv'
_METADATA_FILENAME = 'metadata.csv'


@dataclass
class ExperimentConfig:
    '''Dataclass for storing the experiment configuration.
    '''
    test_size: float
    random_state: int


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


def _make_experiment_filepaths(
    project_name: str,
    data_dir: str | Path,
) -> tuple[Path, str, str, str]:
    data_dir = Path(data_dir)
    split_folder = 'processed'
    basepath = data_dir / project_name / split_folder

    train_filename = 'train.csv'
    test_filename = 'test.csv'
    metadata_filename = 'metadata.csv'

    return basepath, train_filename, test_filename, metadata_filename


def _split_train_test(
    dataset: pd.DataFrame,
    metadata: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    train_dataset, test_dataset = train_test_split(
        dataset,
        test_size=metadata.test_size,
        random_state=metadata.random_state,
    )

    return train_dataset, test_dataset


def _save_dataset(
    dataset: pd.DataFrame,
    filepath: Path,
) -> None:
    dataset.to_csv(filepath, index=False)


def _save_metadata(
    metadata: ExperimentConfig,
    filepath: Path,
) -> None:
    metadata_dict = asdict(metadata)
    with open(filepath, 'w', encoding='utf8') as metadata_file:
        json.dump(metadata_dict, metadata_file, indent=4)


def _save_datasets_and_metadata(
    train_dataset: pd.DataFrame,
    test_dataset: pd.DataFrame,
    metadata: ExperimentConfig,
    data_dir: str | Path,
    project_name: str,
) -> None:
    (
        basepath,
        train_filename,
        test_filename,
        metadata_filename,
    ) = _make_experiment_filepaths(
        data_dir=data_dir,
        project_name=project_name,
    )

    train_filepath = basepath / train_filename
    test_filepath = basepath / test_filename
    metadata_filepath = basepath / metadata_filename

    try:
        basepath.mkdir(parents=True, exist_ok=True)
        _save_dataset(train_dataset, train_filepath)
        _save_dataset(test_dataset, test_filepath)
        _save_metadata(metadata, metadata_filepath)
    except OSError as e:
        print(f'Error saving datasets and metadata: {e}')
        shutil.rmtree(basepath)
        raise e


def split_train_test_and_save(
    dataset: pd.DataFrame,
    metadata: ExperimentConfig,
    data_dir: str | Path,
    project_name: str,
) -> None:
    '''Splits the dataset into train and test sets and saves them to the data_dir.
    '''
    train_dataset, test_dataset = _split_train_test(dataset, metadata)
    _save_datasets_and_metadata(
        train_dataset,
        test_dataset,
        metadata,
        data_dir,
        project_name,
    )


def load_car_dataset_split(
    data_dir: str | Path,
    project_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, ExperimentConfig]:
    '''Loads the train and test datasets and metadata from the data_dir.
    '''
    data_dir = Path(data_dir)

    split_dataset_dir = data_dir / project_name / 'processed'

    train_dataset_path = split_dataset_dir / 'train.csv'
    test_dataset_path = split_dataset_dir / 'test.csv'
    metadata_path = split_dataset_dir / 'metadata.csv'

    train_dataset = pd.read_csv(train_dataset_path)
    test_dataset = pd.read_csv(test_dataset_path)

    with open(metadata_path, 'r', encoding='utf8') as metadata_file:
        metadata_dict = json.load(metadata_file)

    metadata = ExperimentConfig(**metadata_dict)

    return train_dataset, test_dataset, metadata
