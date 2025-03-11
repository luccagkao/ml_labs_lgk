''' JSON template for the stats of numerical and categorical columns. '''
import pandas as pd

_JSON_TEMPLATE = '''
{{
    "numerical": {numerical_stats},
    "categorical": {categorical_stats}
}}
'''


def render_json(
    numerical_stats: pd.DataFrame,
    categorical_stats: pd.DataFrame,
) -> None:
    '''Render the stats of numerical and categorical columns in JSON format.
    '''
    numerical_stats_str = numerical_stats \
        .to_json(indent=4) \
        .replace('\n', '\n    ') \
        .replace(':', ': ')
    categorical_stats_str = categorical_stats \
        .to_json(indent=4) \
        .replace('\n', '\n    ') \
        .replace(':', ': ')
    content = _JSON_TEMPLATE.format(
        numerical_stats=numerical_stats_str,
        categorical_stats=categorical_stats_str,
    )
    return content
