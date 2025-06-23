import pandas as pd
import joblib
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load data
file_path = "Crop_recommendation1.csv"
df = pd.read_csv(file_path)

# Encode target
le = LabelEncoder()
df["label"] = le.fit_transform(df["label"])

# Split input/output
X = df.drop(columns=["label"])
y = df["label"]

# Scale input features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# Train model
start_time = time.time()
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42
)
model.fit(X_train, y_train)
training_time = time.time() - start_time

# Save model, scaler, encoder, and feature names
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(X.columns.tolist(), "features.pkl")

print(f"✅ Model, scaler, encoder, and feature names saved successfully in {training_time:.2f} seconds.") 