''' Markdown template for descriptive statistics '''
import pandas as pd

_MARKDOWN_TEMPLATE = '''
# Descriptive statistics

## Numerical columns

{numerical_stats}

## Categorical columns

{categorical_stats}
'''


def render_markdown(
    numerical_stats: pd.DataFrame,
    categorical_stats: pd.DataFrame,
) -> None:
    '''Render descriptive statistics in markdown format.'''
    numerical_stats_str = numerical_stats.to_markdown()
    categorical_stats_str = categorical_stats.to_markdown()
    content = _MARKDOWN_TEMPLATE.format(
        numerical_stats=numerical_stats_str,
        categorical_stats=categorical_stats_str,
    )
    return content
