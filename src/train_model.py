import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("data.csv")

# Features and labels
X = df[["blink_rate", "ear_mean", "ear_std", "session_time"]]
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
print("\nFeature Importance:")
for name, importance in zip(X.columns, model.feature_importances_):
    print(f"{name}: {importance:.4f}")

import joblib

joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")