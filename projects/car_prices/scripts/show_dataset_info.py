''' Show dataset info '''
import io
from argparse import ArgumentParser
from contextlib import contextmanager

import pandas as pd
from car_prices.dataset import load_car_dataset
from dotenv import dotenv_values
from utils import PRINT_OPTIONS, print_stats


def parse_args() -> dict[str, str]:
    ''' Parse command-line arguments. '''
    parser = ArgumentParser()
    parser.add_argument(
        '-p',
        '--print',
        type=str,
        choices=PRINT_OPTIONS,
        default='text',
        help='Output format',
    )
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        default=None,
        help='Output file',
    )
    return vars(parser.parse_args())


def load_data() -> pd.DataFrame:
    ''' Load the car dataset. '''
    config = dotenv_values()
    data_dir = config['DATA_DIR']
    data = load_car_dataset(data_dir)
    return data


def compute_stats(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ''' Compute dataset statistics. '''
    numerical_stats = data \
        .select_dtypes(include='number') \
        .describe() \
        .transpose()

    categorical_stats = data \
        .select_dtypes(include='object') \
        .describe() \
        .transpose()

    return numerical_stats, categorical_stats


@contextmanager
def get_output(output_option: str) -> io.TextIOBase:
    ''' Get the output file. '''
    if output_option:
        file = open(output_option, 'w', encoding='utf8')
        yield file
        file.close()
    else:
        yield None


def main() -> None:
    ''' Main function. '''
    options = parse_args()
    print_option = options['print']
    output_option = options['output']

    data = load_data()

    numerical_stats, categorical_stats = compute_stats(data)

    with get_output(output_option) as out_file:
        print_stats(numerical_stats, categorical_stats, print_option, out_file)


if __name__ == '__main__':
    main()
