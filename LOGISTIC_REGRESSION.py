import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURATION ---
FILE_NAME = 'student_exam_data.csv' 
FEATURE_COLUMNS = ['Study Hours', 'Previous Exam Score']  # 2 Features
TARGET_COLUMN = 'Pass/Fail'                             # Target (Note: removed list brackets for simpler access)
RANDOM_STATE = 42
TEST_SIZE = 0.3

# --- 1. Load Data ---
try:
    data = pd.read_csv(FILE_NAME)
    print(f"Data loaded successfully. Shape: {data.shape}")
except FileNotFoundError:
    # Create Dummy Data for demonstration if file is missing
    print(f"File '{FILE_NAME}' not found. Generating DUMMY data...")
    np.random.seed(42)
    data = pd.DataFrame({
        'Study Hours': np.random.rand(100) * 10,
        'Previous Exam Score': np.random.randint(40, 100, 100),
        'Pass/Fail': np.random.randint(0, 2, 100)
    })

# --- 2. Prepare Data ---
X = data[FEATURE_COLUMNS]
y = data[TARGET_COLUMN]

# Ensure X is 2D and y is 1D
if X.ndim == 1:
    X = X.values.reshape(-1, 1)
y = y.values.ravel() # Flattens the array to 1D

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=TEST_SIZE, 
    random_state=RANDOM_STATE, 
    stratify=y 
)

# --- 3. Initialize and Train ---
model = LogisticRegression(random_state=RANDOM_STATE) 
print("Training model...")
model.fit(X_train, y_train)
print("Training complete.")

# --- 4. Evaluate ---
y_pred = model.predict(X_test)

print("\n" + "="*50)
print(f"| {'Model Evaluation Results':<48} |")
print("="*50)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("-" * 50)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("-" * 50)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("="*50)

# --- 5. Visualization (Corrected) ---
plt.figure(figsize=(10, 6))

# CASE A: If we have exactly 2 Features -> Draw a Decision Boundary Map
if X.shape[1] == 2:
    print("Plotting Decision Boundary for 2 Features...")
    
    # 1. Create a meshgrid (a rectangular grid of points)
    # Get min and max values for both features with some padding
    x_min, x_max = X[FEATURE_COLUMNS[0]].min() - 1, X[FEATURE_COLUMNS[0]].max() + 1
    y_min, y_max = X[FEATURE_COLUMNS[1]].min() - 1, X[FEATURE_COLUMNS[1]].max() + 1
    
    # Generate the grid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # 2. Predict on the entire grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 3. Plot the contour (background color)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    
    # 4. Plot the actual data points
    scatter = plt.scatter(X[FEATURE_COLUMNS[0]], X[FEATURE_COLUMNS[1]], 
                          c=y, cmap=plt.cm.coolwarm, edgecolors='k', s=80)
    
    plt.xlabel(FEATURE_COLUMNS[0])
    plt.ylabel(FEATURE_COLUMNS[1])
    plt.title('Logistic Regression Decision Boundary (2 Features)')
    
    # Add a legend manually
    plt.legend(handles=scatter.legend_elements()[0], labels=['Fail', 'Pass'])

# CASE B: If we have 1 Feature -> Draw the Sigmoid Curve
elif X.shape[1] == 1:
    print("Plotting Sigmoid Curve for 1 Feature...")
    
    feature_name = FEATURE_COLUMNS[0]
    
    # 1. Create a range of values for the X-axis
    X_range = np.linspace(X[feature_name].min(), X[feature_name].max(), 300).reshape(-1, 1)
    
    # 2. Predict probabilities (the S-curve)
    y_proba = model.predict_proba(X_range)[:, 1]
    
    # 3. Plot points and curve
    plt.scatter(X[y == 0], y[y == 0], color='red', label='Fail (0)', marker='x')
    plt.scatter(X[y == 1], y[y == 1], color='blue', label='Pass (1)', marker='o')
    plt.plot(X_range, y_proba, color='green', linewidth=2, label='Probability Curve')
    
    # 4. Decision Boundary Line
    if model.coef_[0][0] != 0:
        decision_boundary = -model.intercept_[0] / model.coef_[0][0]
        plt.axvline(x=decision_boundary, color='gray', linestyle='--', label='Decision Boundary')
    
    plt.xlabel(feature_name)
    plt.ylabel('Probability of Passing')
    plt.title('Logistic Regression Sigmoid Curve')
    plt.legend()

plt.grid(True, linestyle='--', alpha=0.5)
plt.show()