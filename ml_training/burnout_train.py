import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data_path = os.path.join("../data", "burnout.csv")
df = pd.read_csv(data_path)

# Features & Label
X = df.drop("burnout_risk", axis=1)
y = df["burnout_risk"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Ensure models folder exists
os.makedirs("../models", exist_ok=True)

# Save model
joblib.dump(model, "../models/burnout_model.pkl")

print("🔥 Burnout Model Trained & Saved Successfully!")
