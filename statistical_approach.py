from analysis.utils import number_frequency
import pandas as pd
import numpy as np


def calculate_probabilities(frequency):
    total_draws = frequency.sum()
    probabilities = frequency / total_draws
    return probabilities


data = pd.read_csv('lotto_numbers.csv')
calc_frequency = number_frequency(data)
calc_probabilities = calculate_probabilities(calc_frequency)


def generate_numbers(probabilities, n=6):
    return np.random.choice(probabilities.index, size=n, replace=False, p=probabilities.values)


predicted_numbers = generate_numbers(calc_probabilities)
print("Generated numbers based on historical frequency:", predicted_numbers)
