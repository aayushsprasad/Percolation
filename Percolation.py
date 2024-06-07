import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from openpyxl import Workbook
wb = Workbook()
# Improved Union-Find (Disjoint Set) data structure
class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.size = [1] * size

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            if self.size[root_x] < self.size[root_y]:
                root_x, root_y = root_y, root_x
            self.parent[root_y] = root_x
            self.size[root_x] += self.size[root_y]


def percolation_constant(n):
    grid_size = n * n
    vacant_cells = 0
    grid = np.full((n, n), False, dtype=bool)  # Initially all cells are occupied
    uf = UnionFind(grid_size + 2)  # +2 for the top and bottom plates

    cell_coordinates = [(x, y) for x in range(n) for y in range(n)]
    random.shuffle(cell_coordinates)

    top_plate = grid_size
    bottom_plate = grid_size + 1

    for i in range(n):
        uf.union(top_plate, i)
        uf.union(bottom_plate, grid_size - n + i)

    for i in range(grid_size):
        x, y = cell_coordinates[i]

        if not grid[x][y]:
            grid[x][y] = True
            vacant_cells += 1

            neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
            for nx, ny in neighbors:
                if 0 <= nx < n and 0 <= ny < n and grid[nx][ny]:
                    uf.union(x * n + y, nx * n + ny)

                if uf.find(top_plate) == uf.find(bottom_plate):
                    percolation_probability = vacant_cells / grid_size
                    return percolation_probability

    return vacant_cells / grid_size





# Number of simulations and grid size
num_simulations = 1000
n = 40
results = []
occupancy_fractions = []

for _ in range(num_simulations):
    percolation_prob = percolation_constant(n)
    occupancy_fraction = (n * n) * percolation_prob
    results.append(percolation_prob)
    occupancy_fractions.append(occupancy_fraction)

    print(f"Simulation {len(results)} - Percolation Probability: {percolation_prob:.4f}, Occupancy Fraction: {occupancy_fraction:.2f}")


data = {
    "Simulation": list(range(1, num_simulations + 1)),
    "Percolation Probability": results,
    "Occupancy Fraction": occupancy_fractions
}

df = pd.DataFrame(data)
df.to_excel("percolation_results.xlsx", index=False)
print("data to excel")

# Create the line chart
plt.figure(figsize=(10, 6))
plt.plot(occupancy_fractions, results, marker='o', linestyle='-')
plt.xlabel('Occupancy Fraction')
plt.ylabel('Percolation Probability')
plt.title('Percolation Probability vs Occupancy Fraction')
plt.grid(True)
plt.show()