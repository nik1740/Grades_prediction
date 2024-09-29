import numpy as np
import pandas as pd

# Set a random seed for reproducibility
np.random.seed(42)

# Generate 1000 random data points for hours studied (between 0 and 10 hours)
hours_studied = np.random.uniform(0, 10, 1000)

# Simulate the grades (let's assume a linear relation with some noise)
# Grade = 5 * Hours Studied + 10 + noise
noise = np.random.normal(0, 5, 1000)  # Gaussian noise
grades = 5 * hours_studied + 10 + noise

# Create a DataFrame
data = pd.DataFrame({
    'Hours_Studied': hours_studied,
    'Grades': grades
})

# Save the dataset to a CSV file
data.to_csv('synthetic_student_data.csv', index=False)

print("Dataset created and saved as 'synthetic_student_data.csv'.")
