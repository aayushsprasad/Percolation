import numpy as np
import random
import matplotlib.pyplot as plt


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

    while not uf.find(0) == uf.find(grid_size + 1):  # Grid hasn't percolated yet
        x, y = random.randint(0, n - 1), random.randint(0, n - 1)

        # If the cell is already vacant, skip it
        if not grid[x][y]:
            grid[x][y] = True
            vacant_cells += 1

            # Connect this cell with its vacant neighbors
            neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
            for nx, ny in neighbors:
                if 0 <= nx < n and 0 <= ny < n and grid[nx][ny]:
                    uf.union(x * n + y + 1, nx * n + ny + 1)

    percolation_probability = vacant_cells / grid_size
    return percolation_probability


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

    print(
        f"Simulation {len(results)} - Percolation Probability: {percolation_prob:.4f}, Occupancy Fraction: {occupancy_fraction:.2f}")

# Create a line chart to show the change in percolation probability with increasing occupancy fraction
plt.plot(occupancy_fractions, results, marker='o', linestyle='-')
plt.xlabel('Occupancy Fraction (p * n^2)')
plt.ylabel('Percolation Probability (p)')
plt.title('Percolation Probability vs. Occupancy Fraction')
plt.grid(True)

# Show the plot
plt.show()
