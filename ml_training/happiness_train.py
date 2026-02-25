import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load dataset
data_path = os.path.join("../data", "happiness.csv")
df = pd.read_csv(data_path)

# Features & target
X = df.drop("happiness_score", axis=1)
y = df["happiness_score"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model to /models/
os.makedirs("../models", exist_ok=True)
joblib.dump(model, "../models/happiness_model.pkl")

print("😊 Happiness Regression Model Trained & Saved Successfully!")
