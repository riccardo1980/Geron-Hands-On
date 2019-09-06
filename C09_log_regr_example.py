from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline, make_pipeline

import numpy as np
import warnings 
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    from chap09_resources.regression import LogisticRegression

# fetch dataset
data, target = make_moons(n_samples=500, random_state=42)
[data_train, data_test, target_train, target_test] = train_test_split(data,
                                                                      target,
                                                                      random_state=42)

#### TRAINING
# scale and add constant feature as bias factor
preprocess = make_pipeline(StandardScaler(), PolynomialFeatures(1))
preprocess.fit(data_train)
X_train = preprocess.transform(data_train)

y_train = target_train.reshape(-1,1)

lr = LogisticRegression(n_epochs=200, batch_size=X_train.shape[0])
lr.fit(X_train, y_train)

y_train_pred = lr.predict(X_train)
print('\n TRAIN')
print(classification_report(y_train, y_train_pred))

#### TEST
# scale and add constant feature as bias factor
X_test = preprocess.transform(data_test)

y_test = target_test.reshape(-1,1)

y_test_pred = lr.predict(X_test)
print('\n EVAL')
print(classification_report(y_test, y_test_pred))

best_theta = lr._theta
print('theta:')
print(best_theta)