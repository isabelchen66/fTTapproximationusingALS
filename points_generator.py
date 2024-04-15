import csv
from scipy.spatial import KDTree
import numpy as np

# Load diamond data from CSV file
def load_data_from_csv(file_path):
    data_points = []
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            # Split the row into independent and dependent variables
            variables = [float(val) for val in row[:-1]]
            price = float(row[-1])  # Last element is assumed to be the price
            data_points.append((variables, price))
    return data_points

# File path to your CSV file
csv_file_path = 'diamonds.csv'

# Load data from CSV
data_points = load_data_from_csv(csv_file_path)

# Extract data points for KDTree
coordinates = [point[0] for point in data_points]

# Create a KDTree
kdtree = KDTree(coordinates)

# Example query point (carat, depth, table, length, width)
query_point = (0.23, 61.5, 55, 3.95, 3.98)

# Query nearest neighbors
distance, index = kdtree.query(query_point)

# Retrieve price of nearest neighbor
nearest_neighbor_price = data_points[index][1]

# Print result
print("For a diamond with features (carat, table, length, width, depth) =", query_point)
print("Nearest neighbor price:", nearest_neighbor_price)
# Print result
print("For a diamond with features (carat, table, length, width, depth) =", query_point)
print("Nearest neighbor price:", nearest_neighbor_price)
