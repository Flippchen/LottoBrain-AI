# Description: This script uses a statistical approach to predict the next winning numbers.

# Import libraries
from analysis.utils import number_frequency
import pandas as pd
import numpy as np


# Calculate the probabilities of each number
def calculate_probabilities(frequency: pd.Series) -> pd.Series:
    total_draws = frequency.sum()
    probabilities = frequency / total_draws
    return probabilities


# Generate the next lotto numbers
def generate_numbers(probabilities: pd.Series, n: int = 6) -> np.ndarray:
    return np.random.choice(probabilities.index, size=n, replace=False, p=probabilities.values)


# Load the data from csv file
data: pd.DataFrame = pd.read_csv('../lotto_numbers.csv')
# Calculate the frequency of each number
calc_frequency = number_frequency(data)
# Calculate the probabilities of each number
calc_probabilities = calculate_probabilities(calc_frequency)

# Generate the next lotto numbers
predicted_numbers = generate_numbers(calc_probabilities)
print("Generated numbers based on historical frequency:", predicted_numbers)
