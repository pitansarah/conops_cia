# RUN : python figures1.py

import numpy as np
from sklearn.manifold import MDS


# Data points 
# Grooming: [Consistency, Motivation, Social Use]
data = np.array([
    [8, 8, 9],  # Normal Population
    [5, 5, 3],  # ASPD
    [6, 9, 6],  # NPD
    [4, 3, 2],  # SPD
    [6, 6, 5]   # DID
])


# Empathy
# [Cognitive, Emotional, Compassionate]
empathy_data = np.array([
    [8, 9, 9],  # Normal Population
    [3, 2, 1],  # ASPD
    [5, 6, 4],  # NPD
    [2, 3, 2],  # SPD
    [6, 7, 6]   # DID
])

# Compute Euclidean distance matrix
from scipy.spatial.distance import euclidean
distances = np.array([[euclidean(data[i], data[j]) for j in range(len(data))] for i in range(len(data))])

# Apply MDS
mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42)
mds_coords = mds.fit_transform(distances)

# Normalized coordinates (scale to 0-10 for visualization)
mds_coords = (mds_coords - mds_coords.min()) / (mds_coords.max() - mds_coords.min()) * 10
print(mds_coords)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Scatter with each axis
ax.scatter(mds_coords[:, 0], mds_coords[:, 1], mds_coords[:, 2], c='blue', s=50)

ax.set_xlabel('X (normalized)')
ax.set_ylabel('Y (normalized)')
ax.set_zlabel('Z (normalized)')
ax.set_title('3D MDS Visualization')

plt.show()

# GET 3D MODEL FROM PRINTED COORDINATES
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np

# MDS coordinates
mds_coords = np.array([
    [4.07703128, 10.0, 5.50933974],   # Normal Population
    [5.54289191, 2.18575617, 4.41981333],  # ASPD
    [2.31530957, 6.70814594, 3.86216418],  # NPD
    [6.60723056, 0.0, 5.56969564],    # SPD
    [5.18961784, 4.83817905, 4.37106828]   # DID
])

# Labels and colors
profiles = ['Normal Population', 'ASPD', 'NPD', 'SPD', 'DID']
colors = ['purple', 'red', 'green', 'blue', 'orange']

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(len(profiles)):
    ax.scatter(*mds_coords[i], color=colors[i], label=profiles[i])
    ax.text(*mds_coords[i], profiles[i], fontsize=9)

ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')
ax.set_title('3D MDS Visualization of Psychological Profiles')
ax.legend()

# Not working: plt.show()

plt.savefig("3d_plot.png")
