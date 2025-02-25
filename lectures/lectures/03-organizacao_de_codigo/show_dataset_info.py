# pylint: disable=missing-docstring
import io
from argparse import ArgumentParser
from contextlib import contextmanager
from typing import Callable

import pandas as pd
from car_prices.dataset import load_car_dataset
from dotenv import dotenv_values

PrinterType = Callable[[pd.DataFrame, pd.DataFrame, io.TextIOBase], None]

PRINT_OPTIONS = [
    'markdown',
    'json',
    'text',
]

MARKDOWN_TEMPLATE = '''
# Descriptive statistics

## Numerical columns

{numerical_stats}

## Categorical columns

{categorical_stats}
'''

JSON_TEMPLATE = '''
{{
    "numerical": {numerical_stats},
    "categorical": {categorical_stats}
}}
'''

TEXT_TEMPLATE = '''
Descriptive statistics:

Numerical columns:

{numerical_stats}

Categorical columns:

{categorical_stats}
'''

TEMPLATES = {
    'markdown': MARKDOWN_TEMPLATE,
    'json': JSON_TEMPLATE,
    'text': TEXT_TEMPLATE,
}


def parse_args() -> dict[str, str]:
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
    config = dotenv_values()
    data_dir = config['DATA_DIR']
    data = load_car_dataset(data_dir)
    return data


def compute_stats(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    numerical_stats = data \
        .select_dtypes(include='number') \
        .describe() \
        .transpose()

    categorical_stats = data \
        .select_dtypes(include='object') \
        .describe() \
        .transpose()

    return numerical_stats, categorical_stats


def print_template(
    numerical_stats_str: str,
    categorical_stats_str: str,
    template: str,
    out_file: io.TextIOBase = None,
) -> None:
    print(
        template.format(
            numerical_stats=numerical_stats_str,
            categorical_stats=categorical_stats_str,
        ),
        file=out_file,
    )


def get_template(print_option: str) -> str:
    if print_option not in TEMPLATES:
        print_option = 'text'
    return TEMPLATES[print_option]


@contextmanager
def get_output(output_option: str) -> io.TextIOBase:
    if output_option:
        file = open(output_option, 'w', encoding='utf8')
        yield file
        file.close()
    else:
        yield None


def print_markdown(
    numerical_stats: pd.DataFrame,
    categorical_stats: pd.DataFrame,
    out_file: io.TextIOBase = None,
) -> None:
    numerical_stats_str = numerical_stats.to_markdown()
    categorical_stats_str = categorical_stats.to_markdown()
    template = MARKDOWN_TEMPLATE
    print_template(
        numerical_stats_str,
        categorical_stats_str,
        template,
        out_file,
    )


def print_json(
    numerical_stats: pd.DataFrame,
    categorical_stats: pd.DataFrame,
    out_file: io.TextIOBase = None,
) -> None:
    numerical_stats_str = numerical_stats \
        .to_json(indent=4) \
        .replace('\n', '\n    ') \
        .replace(':', ': ')
    categorical_stats_str = categorical_stats \
        .to_json(indent=4) \
        .replace('\n', '\n    ') \
        .replace(':', ': ')
    template = JSON_TEMPLATE
    print_template(
        numerical_stats_str,
        categorical_stats_str,
        template,
        out_file,
    )


def print_text(
    numerical_stats: pd.DataFrame,
    categorical_stats: pd.DataFrame,
    out_file: io.TextIOBase = None,
) -> None:
    numerical_stats_str = numerical_stats.to_string()
    categorical_stats_str = categorical_stats.to_string()
    template = TEXT_TEMPLATE
    print_template(
        numerical_stats_str,
        categorical_stats_str,
        template,
        out_file,
    )


def get_printer(print_option: str) -> PrinterType:
    printers = {
        'markdown': print_markdown,
        'json': print_json,
        'text': print_text,
    }
    return printers.get(print_option, print_text)


def print_stats(
    numerical_stats: pd.DataFrame,
    categorical_stats: pd.DataFrame,
    options: dict[str, str],
) -> None:
    template_option = options['print']
    output_option = options['output']

    printer = get_printer(template_option)
    with get_output(output_option) as out_file:
        printer(numerical_stats, categorical_stats, out_file)


def main() -> None:
    options = parse_args()
    data = load_data()
    numerical_stats, categorical_stats = compute_stats(data)
    print_stats(numerical_stats, categorical_stats, options)


if __name__ == '__main__':
    main()
