import pandas as pd
from analysis.utils import *

def load_data(file_name):
    data = pd.read_csv(file_name)
    return data


def add_features(data):
    # Add a new feature: sum of the six main numbers
    data['sum_numbers'] = data['number_1'] + data['number_2'] + data['number_3'] + data['number_4'] + data['number_5'] + data['number_6']

    # Add a new feature: average of the six main numbers
    data['avg_numbers'] = data[['number_1', 'number_2', 'number_3', 'number_4', 'number_5', 'number_6']].mean(axis=1)

    return data


def save_data(data, file_name):
    data.to_csv(file_name, index=False)


def main():
    input_file_name = 'lotto_numbers.csv'
    output_file_name = 'lotto_numbers_with_features.csv'

    # Load data
    data = load_data(input_file_name)

    # Add new features
    data = add_features(data)

    # Save the modified data
    save_data(data, output_file_name)


if __name__ == '__main__':
    main()
