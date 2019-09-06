from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline, make_pipeline

import numpy as np
import warnings 
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    from chap09_resources.regression import LinearRegression

def report(y_true, y):
    metrics = [explained_variance_score, max_error, mean_absolute_error, mean_squared_error, r2_score]
    fields = max([ len(m.__name__) for m in metrics])

    for f in metrics:
        print('{:<{}} {: 4.2e}'.format(f.__name__, fields, f(y_true, y)))

# fetch dataset
housing = fetch_california_housing()
[data_train, data_test, target_train, target_test] = train_test_split(housing.data,
                                                                      housing.target,
                                                                      random_state=42)

#### TRAINING
# scale and add constant feature as bias factor
preprocess = make_pipeline(StandardScaler(), PolynomialFeatures(1))
preprocess.fit(data_train)
X_train = preprocess.transform(data_train)

y_train = target_train.reshape(-1,1)

lr = LinearRegression(n_epochs=200, batch_size=X_train.shape[0])
lr.fit(X_train, y_train)

y_train_pred = lr.predict(X_train)
print('\n TRAIN')
report(y_train, y_train_pred)

#### TEST
# scale and add constant feature as bias factor
X_test = preprocess.transform(data_test)

y_test = target_test.reshape(-1,1)

y_test_pred = lr.predict(X_test)
print('\n EVAL')
report(y_test, y_test_pred)

best_theta = lr._theta
print('theta:')
print(best_theta)