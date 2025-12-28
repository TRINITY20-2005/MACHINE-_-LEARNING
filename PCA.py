# -----------------------------------------
# Principal Component Analysis (PCA) using Iris Dataset
# -----------------------------------------

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# -----------------------------------------
# 1. Load the Iris dataset
# -----------------------------------------
iris = load_iris()

# Features
X = iris.data
# Target labels
y = iris.target

# Convert to DataFrame
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y

print("First 5 rows of the Iris Dataset:")
print(df.head())

# -----------------------------------------
# 2. Standardize the feature data
# -----------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------------------
# 3. Apply PCA (Reduce to 2 components)
# -----------------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Add PCA results to DataFrame
df['PC1'] = X_pca[:, 0]
df['PC2'] = X_pca[:, 1]

print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

# -----------------------------------------
# 4. Visualize the PCA Result
# -----------------------------------------

plt.figure(figsize=(8,6))

# Colors for each class
colors = ['red', 'green', 'blue']
labels = iris.target_names

for color, i, label in zip(colors, [0, 1, 2], labels):
     plt.scatter(df.loc[df['target']==i, 'PC1'],
                 df.loc[df['target']==i, 'PC2'],
                 label=label,
alpha=0.7)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Iris Dataset (2 Components)')
plt.legend()
plt.grid(True)
plt.show()