# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# Example dataset
data = {
    'Age': [25, np.nan, 30, 22, 35],
    'Gender': ['Male', 'Female', 'Female', np.nan, 'Male'],
    'Income': [50000, 60000, np.nan, 55000, 65000],
    'City': ['New York', 'Los Angeles', 'New York', 'Chicago', 'Chicago'],
    'Purchased': ['No', 'Yes', 'Yes', 'No', 'Yes']  # target variable
}

df = pd.DataFrame(data)

# Display the original dataframe
print("Original DataFrame:")
print(df)

# Step 1: Handling Missing Data

# Create a transformer to handle missing data in numerical and categorical features
numerical_features = ['Age', 'Income']
categorical_features = ['Gender', 'City']

# Use SimpleImputer to replace missing values
numerical_transformer = SimpleImputer(strategy='mean')  # Replace missing values with the mean
categorical_transformer = SimpleImputer(strategy='most_frequent')  # Replace with the most frequent category

# Step 2: Encoding Categorical Features

# One-hot encoding for categorical variables
categorical_encoder = OneHotEncoder(drop='first')  # Drop one category to avoid multicollinearity

# Step 3: Scaling Numerical Features
numerical_scaler = StandardScaler()  # Standardize numerical features

# Step 4: Combine all preprocessing steps into a single pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_encoder, categorical_features)
    ])

# Step 5: Prepare Target Variable (Encode the target variable)
label_encoder = LabelEncoder()
df['Purchased'] = label_encoder.fit_transform(df['Purchased'])  # Encoding 'Yes' as 1, 'No' as 0

# Step 6: Split the data into features and target
X = df.drop('Purchased', axis=1)  # Features
y = df['Purchased']  # Target variable

# Step 7: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Create a pipeline to combine preprocessing and model training
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    # Optionally, add a model here, e.g., LogisticRegression(), RandomForestClassifier(), etc.
    # ('classifier', LogisticRegression())
])

# Fit the pipeline with the training data
pipeline.fit(X_train, y_train)

# Transform the test data using the same preprocessing steps
X_test_transformed = pipeline.transform(X_test)

# Display the preprocessed data
print("\nPreprocessed Training Data (Transformed):")
print(X_train.head())  # Check preprocessed training data

print("\nTransformed Test Data:")
print(X_test_transformed)

