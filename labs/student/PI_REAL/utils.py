# pylint: disable=missing-docstring
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display, display_markdown
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split


def load_dataset():
    # Load the original dataset.
    data = fetch_covtype(as_frame=True, shuffle=True)
    data = data['frame']

    # Soil type and wilderness area are one-hot encoded.
    # We will combine them into single columns.
    data_soil = data.loc[:, 'Soil_Type_0':'Soil_Type_39']
    data.drop(columns=data_soil.columns, inplace=True)
    data_soil = data_soil.idxmax(axis=1)
    data_soil = data_soil.str \
        .replace('Soil_Type_', '').astype('int')
    data['Soil_Type'] = data_soil.astype('category')

    data_wilderness = data.loc[:, 'Wilderness_Area_0':'Wilderness_Area_3']
    data.drop(columns=data_wilderness.columns, inplace=True)
    data_wilderness = data_wilderness.idxmax(axis=1)
    data_wilderness = data_wilderness.str \
        .replace('Wilderness_Area_', '').astype('int')
    data['Wilderness_Area'] = data_wilderness.astype('category')

    # Cover type is the target variable. Assign human-readable names.
    data['Cover_Type'] = data['Cover_Type'].replace({
        1: 'Spruce/Fir',
        2: 'Lodgepole Pine',
        3: 'Ponderosa Pine',
        4: 'Cottonwood/Willow',
        5: 'Aspen',
        6: 'Douglas-fir',
        7: 'Krummholz',
    }).astype('category')

    return data


def analise_descritiva_preliminar(data):
    # Display the dimensions of the dataset.
    display_markdown('### Numero de linhas e colunas', raw=True)
    display(data.shape)

    # Display the first few rows of the dataset.]
    display_markdown('### Primeiras linhas do dataset', raw=True)
    display(data.head())

    # Display the data types of each column.
    display_markdown('### Tipos de dados de cada coluna', raw=True)
    display(data.dtypes)

    # Display the number of missing values in each column.
    display_markdown('### Numero de valores faltantes em cada coluna', raw=True)
    display(data.isnull().sum())

    display_markdown(
        '### Contagens de valores unicos em cada coluna categorica',
        raw=True,
    )

    # Display the unique values of the target variable.
    display(data['Cover_Type'].value_counts())

    # Display the unique values of the soil type.
    display(data['Soil_Type'].value_counts())

    # Display the unique values of the wilderness area.
    display(data['Wilderness_Area'].value_counts())

    # Display the summary statistics of the quantitative variables.
    display_markdown('### Estatisticas das variáveis numéricas do dataset')
    display(data.select_dtypes(include=['number']).describe().T.round(2))


def visualizacao_preliminar(data):
    aux = data.select_dtypes(include='number')
    for col in aux:
        plt.figure()
        aux[col].plot.hist(bins=67, figsize=(7, 7))
        plt.title(col)
        plt.show()

    aux = data.select_dtypes(include='category')
    for col in aux:
        plt.figure()
        aux[col].value_counts().plot.barh(figsize=(7, 7))
        plt.title(col)
        plt.show()


def split_data(data):
    train_data, test_data = train_test_split(
        data,
        test_size=0.2,
        random_state=42,
        stratify=data[['Soil_Type', 'Cover_Type', 'Wilderness_Area']])
    return train_data, test_data


def plot_hist(data, col):
    plt.figure()
    data[col].plot.hist(bins=67, figsize=(5, 5))
    plt.title(col)
    plt.show()


def plot_bar(data, col):
    plt.figure()
    data[col].value_counts().plot.barh(figsize=(5, 5))
    plt.title(col)
    plt.show()


def plot_num_num(data, col1, col2):
    plt.figure()
    plt.scatter(data[col1], data[col2], alpha=0.1)
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title(f'{col1} vs {col2}')
    plt.show()


def plot_cat_cat(data, col1, col2):
    plt.figure()
    crosstab = pd.crosstab(data[col1], data[col2])
    crosstab = crosstab.div(crosstab.sum(axis=1), axis=0)
    crosstab.plot.bar(stacked=True, figsize=(5, 5))
    plt.title(f'{col1} vs {col2}')
    plt.show()


def plot_num_cat(data, col1, col2):

    def sort_cat_num(data, col1, col2):
        dtype1 = data[col1].dtype.name
        if dtype1 == 'float64':
            col1, col2 = col2, col1
        return col1, col2

    col1, col2 = sort_cat_num(data, col1, col2)
    plt.figure()
    data.boxplot(column=col2, by=col1, figsize=(10, 6))
    plt.title(f'{col1} vs {col2}')
    plt.show()


def plot_single_col(data, col):
    dtype = data[col].dtype.name
    if dtype == 'category':
        plot_bar(data, col)
    else:
        plot_hist(data, col)


def plot_two_cols(data, col1, col2):
    dtype1 = data[col1].dtype.name
    dtype2 = data[col2].dtype.name

    if dtype1 == 'category' and dtype2 == 'category':
        plot_cat_cat(data, col1, col2)
    elif dtype1 == 'float64' and dtype2 == 'float64':
        plot_num_num(data, col1, col2)
    else:
        plot_num_cat(data, col1, col2)


def visualizacao_posterior(train_data):
    for col1 in train_data.columns:
        for col2 in train_data.columns:
            if col1 > col2:
                continue

            if col1 == col2:
                plot_single_col(train_data, col1)
                continue

            plot_two_cols(train_data, col1, col2)
