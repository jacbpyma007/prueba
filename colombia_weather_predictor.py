import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

class ColombiaWeatherPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=150, random_state=42, max_depth=15)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.regions = {
            'caribbean': {'lat': 10.5, 'elevation': 10},
            'pacific': {'lat': 3.0, 'elevation': 50},
            'andes': {'lat': 5.0, 'elevation': 2500},
            'amazon': {'lat': -1.0, 'elevation': 100},
            'llanos': {'lat': 4.5, 'elevation': 150}
        }

    def generate_synthetic_data(self, n_samples=1000):
        """Generate synthetic weather data representative of Colombia"""
        np.random.seed(42)
        
        # Regional distribution
        regions = np.random.choice(list(self.regions.keys()), n_samples)
        
        # Features based on Colombian geography and climate
        latitude = np.array([self.regions[r]['lat'] for r in regions]) + np.random.normal(0, 2, n_samples)
        elevation = np.array([self.regions[r]['elevation'] for r in regions]) + np.random.normal(0, 300, n_samples)
        elevation = np.clip(elevation, 0, 5000)  # Colombia max elevation ~5000m
        
        # Humidity (tropical country, generally high)
        humidity = np.random.uniform(60, 95, n_samples)
        
        # Precipitation (rainy seasons in Colombia)
        precipitation = np.random.uniform(100, 500, n_samples)
        
        # Wind speed (generally moderate)
        wind_speed = np.random.uniform(2, 15, n_samples)
        
        # Season indicator (1=rainy, 0=dry)
        season = np.random.uniform(0, 1, n_samples)
        
        # Temperature model for Colombia:
        # - Higher elevation = lower temperature (approx -0.6°C per 100m)
        # - Tropical baseline ~25-28°C
        # - Humidity and precipitation affect temperature
        temperature = (28 - 0.006*elevation + 0.15*humidity - 0.02*precipitation 
                      + 0.3*wind_speed - 2*season + np.random.normal(0, 1.5, n_samples))
        
        X = np.column_stack([latitude, elevation, humidity, precipitation, wind_speed, season])
        y = temperature
        
        return X, y

    def train(self, X=None, y=None):
        """Train the model with provided data or generate synthetic Colombian data"""
        if X is None or y is None:
            X, y = self.generate_synthetic_data()
        
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
        
        print(f"Colombia Weather Model trained successfully!")
        print(f"MSE: {mse:.2f}, R² Score: {r2:.4f}")
        return mse, r2

    def predict(self, latitude, elevation, humidity, precipitation, wind_speed, season):
        """
        Predict temperature for Colombian location
        
        Args:
            latitude: Geographic latitude
            elevation: Elevation in meters
            humidity: Humidity percentage (60-95 typical for Colombia)
            precipitation: Monthly precipitation in mm
            wind_speed: Wind speed in km/h
            season: Season indicator (0=dry, 1=rainy)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        features = np.array([[latitude, elevation, humidity, precipitation, wind_speed, season]])
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)
        return prediction[0]

    def predict_by_region(self, region_name, humidity, precipitation, wind_speed, season):
        """Predict temperature for a known Colombian region"""
        if region_name not in self.regions:
            raise ValueError(f"Region must be one of: {list(self.regions.keys())}")
        
        region = self.regions[region_name]
        return self.predict(region['lat'], region['elevation'], humidity, precipitation, wind_speed, season)

    def save_model(self, filepath="colombia_weather_model.pkl"):
        """Save trained model to disk"""
        joblib.dump(self.model, filepath)
        joblib.dump(self.scaler, filepath.replace('.pkl', '_scaler.pkl'))
        print(f"Model saved to {filepath}")

    def load_model(self, filepath="colombia_weather_model.pkl"):
        """Load model from disk"""
        self.model = joblib.load(filepath)
        self.scaler = joblib.load(filepath.replace('.pkl', '_scaler.pkl'))
        self.is_trained = True
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    predictor = ColombiaWeatherPredictor()
    predictor.train()
    
    # Prediction examples for different Colombian regions
    print("\n--- Colombian Weather Predictions ---")
    
    # Caribbean coast (Cartagena-like)
    temp_caribbean = predictor.predict_by_region('caribbean', humidity=85, precipitation=150, wind_speed=10, season=0)
    print(f"Caribbean coast prediction: {temp_caribbean:.2f}°C")
    
    # Andes mountains (Bogotá-like)
    temp_andes = predictor.predict_by_region('andes', humidity=70, precipitation=200, wind_speed=8, season=1)
    print(f"Andes mountains prediction: {temp_andes:.2f}°C")
    
    # Amazon region (Leticia-like)
    temp_amazon = predictor.predict_by_region('amazon', humidity=90, precipitation=350, wind_speed=5, season=1)
    print(f"Amazon region prediction: {temp_amazon:.2f}°C")
