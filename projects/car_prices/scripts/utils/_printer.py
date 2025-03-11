''' Module for printing dataset statistics. '''
import io
from typing import Callable

import pandas as pd

from ._json import render_json
from ._markdown import render_markdown
from ._text import render_text

PrinterType = Callable[[pd.DataFrame, pd.DataFrame, io.TextIOBase], None]
RendererType = Callable[[pd.DataFrame, pd.DataFrame], str]

PRINT_OPTIONS = [
    'markdown',
    'json',
    'text',
]

_renderers = {
    'markdown': render_markdown,
    'json': render_json,
    'text': render_text,
}


def _get_renderer(print_option: str) -> RendererType:
    if print_option == 'markdown':
        return render_markdown
    if print_option == 'json':
        return render_json
    return render_text


def print_stats(
    numerical_stats: pd.DataFrame,
    categorical_stats: pd.DataFrame,
    print_option: str = 'text',
    out_file: io.TextIOBase = None,
) -> None:
    '''Print dataset statistics.'''
    renderer = _get_renderer(print_option)
    content = renderer(numerical_stats, categorical_stats)
    print(content, file=out_file)
