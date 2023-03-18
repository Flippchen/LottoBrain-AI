# Import necessary helper functions
from utils import *


# Load lotto_numbers.csv file into a Pandas DataFrame
data = pd.read_csv("../lotto_numbers.csv")

# Calculate the frequency of each number in the lottery results
frequency = number_frequency(data)
print("Frequency of each number:\n", frequency)

# Find the most and least frequent numbers from the frequency dictionary
most_frequent, least_frequent = most_and_least_frequent_numbers(frequency)
print("Most frequent number:", most_frequent)
print("Least frequent number:", least_frequent)

# Calculate the average super_number, excluding -1
average_super_number = average_super_number(data)
print("Average super_number (excluding -1):", average_super_number)

# Plot a bar chart of the frequency of each number
plot_frequency(frequency)

# Create a new column in the DataFrame with the sorted numbers for each draw
data['sorted_numbers'] = data.apply(sorted_numbers, axis=1)
# Count the frequency of each sequence of numbers in the sorted_numbers column
sequence_counts = data['sorted_numbers'].value_counts()
# Find the most frequent sequence of numbers and its count
most_frequent_seq, count = most_frequent_sequence(sequence_counts)
print("Most frequent sequence of numbers:", most_frequent_seq)
print("Number of occurrences:", count)

# Calculate the frequency of numbers in each position in the lottery results
position_frequencies = position_frequency(data)
# Print the frequency of numbers in each position (1-6) in the lottery results
for position, frequency in position_frequencies.items():
    print(f"Frequency of numbers in {position}:\n{frequency}\n")

# Plot the frequency of numbers in each position in a stacked bar chart
plot_position_frequencies(position_frequencies)

# Calculate the time difference between each occurrence of each number in the lottery results
time_diffs = time_between_occurrences(data)

# Calculate the average time between occurrences for each number
avg_times = avg_time_between_occurrences(time_diffs)
print("Average time between occurrences for each number:\n", avg_times)

# Count the frequency of each pair of numbers in the lottery results
pair_counts = count_number_pairs(data)
# Find the most and least common pairs of numbers and print them
most_common_pair, least_common_pair = most_least_common_pairs(pair_counts)
print("Most common pair of numbers:", most_common_pair)
print("Least common pair of numbers:", least_common_pair)

# Plot a histogram of the sum of each draw in the lottery results
sum_distribution(data)

# Plot a boxplot of the sum of each decade in the lottery results
boxplot_sum_by_decade(data)
# Plot a heatmap of the frequency of each pair of numbers in the lottery results
heatmap_number_pairs(pair_counts)
