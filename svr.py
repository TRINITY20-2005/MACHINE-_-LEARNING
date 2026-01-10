import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error

# dataset form kaggle
df=pd.read_csv('Housing.csv')

#columbs
categorical_cols = ['price','area','bedrooms','bathrooms','stories','mainroad','guestroom','basement','hotwaterheating','airconditioning','parking','prefarea','furnishingstatus']

df_processed = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
# Separate Features (X) and Target (y)
X = df_processed.drop('price', axis=1)
y = df_processed['price']

print("Data processed. Features shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)