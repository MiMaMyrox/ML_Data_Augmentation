## TODO Create small pipeline that makes prediction
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

STANDARD_MODEL_ARGS = {
    "n_estimators":100, 
    "random_state":76344
}


class BaseLineModel():
    def __init__(self, **kwargs):
        if kwargs is None:
            kwargs = STANDARD_MODEL_ARGS
        self.pipeline = Pipeline([
            ("min_max_scaler", MinMaxScaler()),
            ("regressor", RandomForestRegressor(**kwargs))
        ])

    def fit(self, X, y):
        self.pipeline.fit(X, y)

    def predict(self, X, y=None):
        return self.pipeline.predict(X)
