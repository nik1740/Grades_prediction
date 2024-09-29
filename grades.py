import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# Step 1: Load Data from a CSV File
csv_file = 'synthetic_student_data.csv'
df = pd.read_csv(csv_file)

# Step 2: Prepare Features and Target
X = df[['Hours_Studied']]  # Features (independent variable)
y = df['Grades']  # Target (dependent variable)

# Step 3: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Step 5: Make Predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the Model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")

# Step 7: Visualize the Correlation
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.title('Hours Studied vs. Grades')
plt.xlabel('Hours Studied')
plt.ylabel('Grades')
plt.legend()
plt.show()

# Step 8: Predict for New Data
new_hours = np.array([[9]])  # Example: Predicting for 9 hours studied
predicted_score = model.predict(new_hours)
print(f"Predicted Grade for 9 hours of study: {predicted_score[0]}")
