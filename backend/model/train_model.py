import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os

# ✅ Load your dataset
data_path = os.path.join(os.path.dirname(__file__), '../data/soil_data.csv')
df = pd.read_csv(data_path)

# ✅ Features and target
X = df[['moisture', 'ph', 'nitrogen', 'phosphorus', 'potassium']]
y = df['soil_health_index']

# ✅ Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ Evaluate
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
print(f"Model trained with MSE: {mse:.2f}")

# ✅ Save model
model_path = os.path.join(os.path.dirname(__file__), 'soil_health_model.pkl')
joblib.dump(model, model_path)
print("Model saved to", model_path)
