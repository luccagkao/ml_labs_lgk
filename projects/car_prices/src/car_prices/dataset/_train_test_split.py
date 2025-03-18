'''Module for splitting the dataset into train and test datasets.
'''
import pandas as pd
from sklearn.model_selection import train_test_split


def split_train_test(
    dataset: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ''' Splits the dataset into train and test datasets. '''

    train_dataset, test_dataset = train_test_split(
        dataset,
        test_size=test_size,
        random_state=random_state,
    )

    return train_dataset, test_dataset
