import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load the dataset
df = pd.read_csv("C:\\Users\\syedu\\Desktop\\IMP\\AirQualityPrediction (Cleaned).csv")  # Replace with your file name

# Encoding categorical features
label_encoders = {}
categorical_columns = ["Pollutant Type", "Country", "Emirate", "Station Name", "Station Location Type"]

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features and target
X = df[["Year", "Pollutant Type", "Country", "Emirate", "Station Name", "Station Location Type"]]
y = df["Value"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
with open("air_quality_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save the label encoders for later use
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

print("Model and label encoders saved successfully!")
