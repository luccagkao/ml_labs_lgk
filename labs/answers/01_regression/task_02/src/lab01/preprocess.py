'''Pre-processes the California Housing Prices dataset.
'''
import numpy as np
import pandas as pd


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    '''Pre-processes the California Housing Prices dataset.

    Pre-processes the California Housing Prices dataset by removing duplicates
    and filtering out invalid rows.

    Args:
        data: A pandas DataFrame containing the California Housing Prices dataset.

    Returns:
        A pandas DataFrame containing the pre-processed California Housing Prices dataset.
    '''
    # Remove duplicates.
    data = data.drop_duplicates()

    # Remove rows with spikes.
    valid_rows = ((data['median_income'] < 15) &
                  (data['housing_median_age'] < 52) &
                  (data['median_house_value'] < 500001))

    # Remove rows with ocean_proximity == 'ISLAND'.
    valid_rows &= data['ocean_proximity'] != 'ISLAND'

    data = data[valid_rows]

    # Compute new features.
    data['rooms_per_household'] = data['total_rooms'] / data['households']
    data['bedrooms_per_room'] = data['total_bedrooms'] / data['total_rooms']
    data['population_per_household'] = data['population'] / data['households']

    data = data.drop(columns=['total_rooms', 'total_bedrooms', 'population'])

    # Apply log transformation to selected features.
    scale_features = [
        'households',
        'median_income',
        'rooms_per_household',
        'population_per_household',
        'bedrooms_per_room',
        'median_house_value',
    ]

    for feature in scale_features:
        data[f'log_{feature}'] = data[feature].map(np.log10)

    data = data.drop(columns=scale_features)

    # Crop outliers.
    log_cut_point = 2.0
    valid_rows = data['log_households'] > log_cut_point
    data = data[valid_rows]

    return data
