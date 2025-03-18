''' This module contains functions to load the car dataset. '''
from ._metadata import ExperimentConfig, load_metadata, save_metadata
from ._raw_dataset_loader import load_car_dataset
from ._train_test_datasets import load_datasets, save_datasets
from ._train_test_split import split_train_test
from ._dataset import split_train_test_and_save, load_car_dataset_split
