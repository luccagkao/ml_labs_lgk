'''Text template for dataset statistics.'''
import pandas as pd

_TEXT_TEMPLATE = '''
Descriptive statistics:

Numerical columns:

{numerical_stats}

Categorical columns:

{categorical_stats}
'''


def render_text(
    numerical_stats: pd.DataFrame,
    categorical_stats: pd.DataFrame,
) -> str:
    '''Render dataset statistics as text.'''
    numerical_stats_str = numerical_stats.to_string()
    categorical_stats_str = categorical_stats.to_string()
    content = _TEXT_TEMPLATE.format(
        numerical_stats=numerical_stats_str,
        categorical_stats=categorical_stats_str,
    )
    return content
