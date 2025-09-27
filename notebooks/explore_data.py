import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

df = pd.read_csv('data/landslide.csv')

# Check columns again
print("Columns in dataset:", df.columns.tolist())

# Drop missing values
df = df.dropna()

features = [
    'Rainfall_mm',
    'Slope_Angle',
    'Soil_Saturation',
    'Vegetation_Cover',
    'Earthquake_Activity',
    'Proximity_to_Water',
    'Soil_Type_Gravel',
    'Soil_Type_Sand',
    'Soil_Type_Silt'
]

target = 'Landslide'

# Check if all features are present
missing_cols = [col for col in features if col not in df.columns]
if missing_cols:
    raise Exception(f"Missing columns in dataset: {missing_cols}")

X = df[features]  # All features numeric or already one-hot encoded
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model and feature columns
joblib.dump(model, 'model/landslide_model.pkl')
joblib.dump(features, 'model/feature_columns.pkl')

print("Model and feature columns saved successfully.")
