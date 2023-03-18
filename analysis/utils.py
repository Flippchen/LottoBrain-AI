from typing import Hashable, Tuple, Any, Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def number_frequency(data: pd.DataFrame) -> pd.Series:
    numbers_columns = ['number_1', 'number_2', 'number_3', 'number_4', 'number_5', 'number_6']
    frequency = data[numbers_columns].stack().value_counts()
    return frequency


def most_and_least_frequent_numbers(frequency: pd.Series) -> Tuple[Hashable, Hashable]:
    most_frequent = frequency.idxmax()
    least_frequent = frequency.idxmin()
    return most_frequent, least_frequent


def average_super_number(data: pd.DataFrame) -> float:
    super_number_data = data['super_number']
    average = super_number_data[super_number_data != -1].mean()
    return average


def plot_frequency(frequency: pd.Series) -> None:
    plt.figure(figsize=(14, 6))
    plt.bar(frequency.index, frequency.values)
    plt.xlabel('Number')
    plt.ylabel('Frequency')
    plt.title('Frequency of Lotto Numbers')
    fig = plt.gcf()
    plt.show()
    fig.savefig('images/number_frequency.png')


def sorted_numbers(row: pd.Series) -> Tuple[Any, ...]:
    numbers = row[['number_1', 'number_2', 'number_3', 'number_4', 'number_5', 'number_6']].tolist()
    return tuple(sorted(numbers))


def most_frequent_sequence(sequence_counts: pd.Series) -> Tuple[Hashable, Any]:
    most_frequent_seq = sequence_counts.idxmax()
    count = sequence_counts.max()
    return most_frequent_seq, count


def position_frequency(data: pd.DataFrame) -> dict[str, pd.Series]:
    position_frequencies = {}
    for column in ['number_1', 'number_2', 'number_3', 'number_4', 'number_5', 'number_6']:
        position_frequencies[column] = data[column].value_counts()
    return position_frequencies


def plot_position_frequencies(position_frequencies: dict[str, pd.Series]) -> None:
    num_positions = len(position_frequencies)
    plt.figure(figsize=(16, num_positions * 4))

    for i, (position, frequency) in enumerate(position_frequencies.items(), start=1):
        plt.subplot(num_positions, 1, i)
        plt.bar(frequency.index, frequency.values)
        plt.xlabel('Number')
        plt.ylabel('Frequency')
        plt.title(f'Frequency of Numbers in {position}')

    plt.tight_layout()
    fig = plt.gcf()
    plt.show()
    fig.savefig('images/position_frequencies.png')


def time_between_occurrences(data: pd.DataFrame) -> dict[int, pd.Series]:
    time_diffs = {}
    for num in range(1, 50):
        occurrences = data[data.apply(lambda row: num in row[['number_1', 'number_2', 'number_3', 'number_4', 'number_5', 'number_6']].values, axis=1)].copy()
        occurrences['date'] = pd.to_datetime(occurrences['date'])
        time_diff = occurrences['date'].diff().dt.days[1:]
        time_diffs[num] = time_diff
    return time_diffs


def avg_time_between_occurrences(time_diffs: dict[int, pd.Series]) -> dict[int, float]:
    avg_times = {num: diff.mean() for num, diff in time_diffs.items()}
    return avg_times


def count_number_pairs(data: pd.DataFrame) -> dict[Tuple[int, int], int]:
    pair_counts = {}
    for index, row in data.iterrows():
        numbers = sorted(row[['number_1', 'number_2', 'number_3', 'number_4', 'number_5', 'number_6']].tolist())
        for i, num1 in enumerate(numbers[:-1]):
            for num2 in numbers[i + 1:]:
                pair = (num1, num2)
                if pair not in pair_counts:
                    pair_counts[pair] = 0
                pair_counts[pair] += 1
    return pair_counts


def most_least_common_pairs(pair_counts: Dict[Tuple[int, int], int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    most_common_pair = max(pair_counts, key=pair_counts.get)
    least_common_pair = min(pair_counts, key=pair_counts.get)
    return most_common_pair, least_common_pair


def sum_distribution(data: pd.DataFrame) -> None:
    data['sum'] = data[['number_1', 'number_2', 'number_3', 'number_4', 'number_5', 'number_6']].sum(axis=1)
    plt.figure(figsize=(12, 6))
    plt.hist(data['sum'], bins=50, edgecolor='black')
    plt.xlabel('Sum of Numbers')
    plt.ylabel('Frequency')
    plt.title('Distribution of the Sum of Numbers in Each Drawing')
    fig = plt.gcf()
    plt.show()
    fig.savefig('images/sum_distribution.png')


def boxplot_sum_by_decade(data: pd.DataFrame) -> None:
    data['decade'] = (data['date'].str[:4].astype(int) // 10) * 10
    data['sum'] = data[['number_1', 'number_2', 'number_3', 'number_4', 'number_5', 'number_6']].sum(axis=1)

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='decade', y='sum', data=data)
    plt.xlabel('Decade')
    plt.ylabel('Sum of Numbers')
    plt.title('Box Plot of the Sum of Numbers in Each Drawing by Decade')
    fig = plt.gcf()
    plt.show()
    fig.savefig('images/boxplot_sum_by_decade.png')


def heatmap_number_pairs(pair_counts: pd.Series) -> None:
    pair_matrix = np.zeros((49, 49))

    for (num1, num2), count in pair_counts.items():
        pair_matrix[num1 - 1, num2 - 1] = count
        pair_matrix[num2 - 1, num1 - 1] = count

    plt.figure(figsize=(12, 10))
    sns.heatmap(pair_matrix, cmap='coolwarm', annot=True, fmt='.0f', cbar_kws={'label': 'Pair Frequency'})
    plt.xlabel('Number 1')
    plt.ylabel('Number 2')
    plt.title('Heatmap of Number Pair Frequencies')
    fig = plt.gcf()
    plt.show()
    fig.savefig('images/pair_heatmap.png')
