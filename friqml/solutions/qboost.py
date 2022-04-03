import pennylane as qml
from pennylane import numpy as np
import dimod
from sklearn.ensemble import RandomForestClassifier
metric = sklearn.metrics.accuracy_score


# EXERCISE 1
def train_rf_model(X_train, y_train):
    return RandomForestClassifier(n_estimators=20, random_state=0).fit(X_train, y_train)


# EXERCISE 2
def get_qubo(predictions, n_models, y_train, lambd):
    q = predictions @ predictions.T/(n_models ** 2)

    qii = len(y_train) / (n_models ** 2) + lambd - \
        2 * predictions @ y_train/(n_models)

    q[np.diag_indices_from(q)] = qii
    Q = {}
    for i in range(n_models):
        for j in range(i, n_models):
            Q[(i, j)] = q[i, j]
    return Q


# EXERCISE 3
def predict(models, weights, X):
    n_data = len(X)
    T = 0
    y = np.zeros(n_data)
    for i, h in enumerate(models):
        y0 = weights[i] * (2*h.predict(X)-1)  # prediction of weak classifier
        y += y0
        T += np.sum(y0)
    y = np.sign(y - T / (n_data*len(models)))
    return y


def solve_lambda(predictions, rf_model, y_train, lam):
    Q = get_qubo(predictions, len(rf_model.estimators_), y_train, lam)
    sampler = dimod.SimulatedAnnealingSampler()
    response = sampler.sample_qubo(Q, num_reads=10)
    weights = list(response.first.sample.values())
    return weights


def find_weights(predictions, rf_model, X_train, y_train, rtol=0.005):
    models = rf_model.predict(X_train)
    n = len(models)
    ws = np.ones(n)
    acc0 = metric(y_train, models)
    acc = acc0
    ls = 0
    for lam in np.arange(12, 25, 0.5):
        weights = solve_lambda(predictions, rf_model, y_train, lam)
        acc1 = metric(y_train, predict(rf_model.estimators_, weights, X_train))
        adiff = abs(acc0-acc1)
        if adiff < rtol:
            acc = acc1
            ws = weights
            ls = lam
        print(lam, acc, np.sum(weights))
    return ws
