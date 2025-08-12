import numpy as np
import plotly.graph_objects as go

# Define the data
data = np.array([
    [8, 8, 9],  # Normal Population - Grooming: [Consistency, Motivation, Social Influence]
    [5, 5, 3],  # ASPD
    [6, 9, 6],  # NPD
    [4, 3, 2],  # SPD
    [6, 6, 5]   # DID
])

empathy_data = np.array([
    [8, 9, 9],  # Normal Population - Empathy: [Cognitive, Emotional, Compassionate]
    [3, 2, 1],  # ASPD
    [5, 6, 4],  # NPD
    [2, 3, 2],  # SPD
    [6, 7, 6]   # DID
])

labels = ['Normal Population', 'ASPD', 'NPD', 'SPD', 'DID']

# Concatenate to create 6D data
combined_data = np.hstack((data, empathy_data))  # Shape: (5, 6)

# PCA: Center the data
mean = np.mean(combined_data, axis=0)
centered = combined_data - mean

# Compute covariance matrix
cov = np.cov(centered.T)

# Compute eigenvalues and eigenvectors
eigvals, eigvecs = np.linalg.eig(cov)

# Sort eigenvectors by descending eigenvalues
idx = np.argsort(eigvals)[::-1]
eigvecs = eigvecs[:, idx]
projected = centered @ eigvecs[:, :3]  # Project to 3D: (5, 3)

# Create the interactive 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=projected[:, 0],
    y=projected[:, 1],
    z=projected[:, 2],
    mode='markers+text',
    marker=dict(
        size=10,
        color=np.arange(len(labels)),  # Color by group index
        colorscale='Viridis',
        opacity=0.8
    ),
    text=labels,
    textposition='top center'
)])

# Customize layout
fig.update_layout(
    title='Interactive 3D Model: PCA Projection of Grooming + Empathy Data',
    scene=dict(
        xaxis_title='Principal Component 1',
        yaxis_title='Principal Component 2',
        zaxis_title='Principal Component 3',
        aspectmode='cube'
    ),
    width=800,
    height=600
)

# Show the plot
fig.show()
# Optional: Save as HTML
fig.write_html('3d_interactive_model.html')