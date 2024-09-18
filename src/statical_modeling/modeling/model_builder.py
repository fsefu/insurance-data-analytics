from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression



class ModelBuilder:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.models = {}
        self.results = {}


    def train_linear_regression(self):
        # Drop rows with missing values
        X_train_clean = self.X_train.dropna()
        y_train_clean = self.y_train[self.X_train.index.isin(X_train_clean.index)]
        
        lr_model = LinearRegression()
        lr_model.fit(X_train_clean, y_train_clean)
        self.models['Linear Regression'] = lr_model


        
    def train_random_forest(self):
        """Train a Random Forest model."""
        rf_model = RandomForestRegressor(random_state=42)
        rf_model.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = rf_model

    def train_xgboost(self):
        """Train an XGBoost model."""
        xgb_model = xgb.XGBRegressor(random_state=42)
        xgb_model.fit(self.X_train, self.y_train)
        self.models['XGBoost'] = xgb_model

    def evaluate_models(self):
        """Evaluate the models and store the results."""
        for name, model in self.models.items():
            predictions = model.predict(self.X_test)
            mse = mean_squared_error(self.y_test, predictions)
            r2 = r2_score(self.y_test, predictions)
            self.results[name] = {'MSE': mse, 'R2 Score': r2}

    def display_evaluation(self):
        """Print evaluation metrics for each model."""
        for name, metrics in self.results.items():
            print(f"{name} - MSE: {metrics['MSE']:.4f}, R2 Score: {metrics['R2 Score']:.4f}")
