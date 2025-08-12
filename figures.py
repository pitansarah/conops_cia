# Run: python figures.py


# Sample data: distances between 5 points

# Array Parameters (L to R): 1. grooming | 2. empathy| 3. eating habits | 4. spirituality | 5. relationships | 6. intelligence | 7. reality testing | 8. positive symptoms | 9. financial habits | 10. clairvoyance

import pandas as pd
import matplotlib.pyplot as plt

# Data points (x, y) and corresponding labels
points = [
    [3, 1],  # ASPD: Aggressive, Hostile
    [9, 2],  # ASPD: Manipulative, Deceitful
    [4, 3],  # ASPD: Impulsive, Risk-Seeking
    [7, 1],  # ASPD: Callous, Unemotional
    [0, 4],  # SPD: Schizophrenia
    [3, 5],  # SPD: Schizophreniform
    [2, 5],  # SPD: Schizoaffective Disorder
    [5, 6],  # SPD: Delusional Disorder
    [6, 6],  # SPD: Brief Psychotic Disorder
    [4, 5],  # SPD: Schizotypal Personality Disorder
    [1, 6],  # SPD: Undifferentiated
    [8, 2],  # NPD: Grandiose, Overt
    [6, 4],  # NPD: Vulnerable, Covert
    [9, 1],  # NPD: Malignant, Toxic
    [6, 8],  # General Population
]

labels = [
    "ASPD: Aggressive/Hostile",
    "ASPD: Manipulative/Deceitful",
    "ASPD: Impulsive/Risk-Seeking",
    "ASPD: Callous/Unemotional",
    "SPD: Schizophrenia",
    "SPD: Schizophreniform",
    "SPD: Schizoaffective",
    "SPD: Delusional Disorder",
    "SPD: Brief Psychotic Disorder",
    "SPD: Schizotypal PD",
    "SPD: Undifferentiated",
    "NPD: Grandiose/Overt",
    "NPD: Vulnerable/Covert",
    "NPD: Malignant/Toxic",
    "General Population",
]

# Create DataFrame
df = pd.DataFrame(points, columns=["x", "y"])
df["label"] = labels

# Plot
plt.figure(figsize=(10, 7))
plt.scatter(df["x"], df["y"], s=50, c="blue", alpha=0.7, edgecolors="black")
plt.title("MDS-style Plot of Psychiatric Categories", fontsize=14)
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')

# Add annotations
for i, row in df.iterrows():
    plt.annotate(row['label'], (row["x"] + 0.08, row["y"] + 0.08), fontsize=9)

# Save plot
plt.savefig("mds_plot.png", bbox_inches="tight", dpi=150)
plt.show()
