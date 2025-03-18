'''Module for loading the car dataset.
'''
import json
from dataclasses import asdict, dataclass
from pathlib import Path

_METADATA_FILENAME = 'metadata.csv'


@dataclass
class ExperimentConfig:
    '''Dataclass for storing the experiment configuration.
    '''
    test_size: float
    random_state: int


def save_metadata(
    metadata: ExperimentConfig,
    basepath: Path,
) -> None:
    '''Saves the experiment configuration to the data_dir.
    '''
    filepath = basepath / _METADATA_FILENAME
    metadata_dict = asdict(metadata)
    with open(filepath, 'w', encoding='utf8') as metadata_file:
        json.dump(metadata_dict, metadata_file, indent=4)


def load_metadata(basepath: Path,) -> ExperimentConfig:
    '''Loads the experiment configuration from the data_dir.
    '''
    filepath = basepath / _METADATA_FILENAME
    with open(filepath, 'r', encoding='utf8') as metadata_file:
        metadata_dict = json.load(metadata_file)
    metadata = ExperimentConfig(**metadata_dict)
    return metadata
