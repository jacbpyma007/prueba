import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

class WeatherPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False

    def generate_sample_data(self, n_samples=500):
        """Generate sample weather data for demonstration"""
        np.random.seed(42)
        humidity = np.random.uniform(30, 95, n_samples)
        pressure = np.random.uniform(1000, 1030, n_samples)
        wind_speed = np.random.uniform(0, 30, n_samples)
        cloud_cover = np.random.uniform(0, 100, n_samples)
        
        # Temperature influenced by these features
        temperature = (15 + 0.3*humidity - 0.01*pressure + 0.5*wind_speed 
                      - 0.05*cloud_cover + np.random.normal(0, 2, n_samples))
        
        X = np.column_stack([humidity, pressure, wind_speed, cloud_cover])
        y = temperature
        return X, y

    def train(self, X=None, y=None):
        """Train the model with provided data or generate sample data"""
        if X is None or y is None:
            X, y = self.generate_sample_data()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        y_pred = self.model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model trained successfully!")
        print(f"MSE: {mse:.2f}, R² Score: {r2:.2f}")
        return mse, r2

    def predict(self, humidity, pressure, wind_speed, cloud_cover):
        """Predict temperature for given weather features"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        features = np.array([[humidity, pressure, wind_speed, cloud_cover]])
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)
        return prediction[0]

    def save_model(self, filepath="weather_model.pkl"):
        """Save trained model to disk"""
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath="weather_model.pkl"):
        """Load model from disk"""
        self.model = joblib.load(filepath)
        self.is_trained = True
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    predictor = WeatherPredictor()
    predictor.train()
    
    temp = predictor.predict(humidity=75, pressure=1013, wind_speed=10, cloud_cover=50)
    print(f"\nPredicted temperature: {temp:.2f}°C")
