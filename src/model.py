from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from config import RANDOM_SEED, RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_MIN_SAMPLES_SPLIT, RF_MIN_SAMPLES_LEAF
import joblib


class VolatilityModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS,
                                           max_depth=RF_MAX_DEPTH,
                                           min_samples_split=RF_MIN_SAMPLES_SPLIT,
                                           min_samples_leaf=RF_MIN_SAMPLES_LEAF,
                                           random_state=RANDOM_SEED)

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def evaluate(self, x_test, y_test):
        predictions = self.model.predict(x_test)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        return {'MSE': mse, 'MAE': mae}

    def predict(self, x):
        return self.model.predict(x)

    def feature_importances(self):
        return self.model.feature_importances_

    def save(self, file_path):
        joblib.dump(self.model, file_path)

    def load(self, file_path):
        self.model = joblib.load(file_path)
