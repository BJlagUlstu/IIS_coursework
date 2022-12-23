from sklearn.metrics import mean_absolute_percentage_error
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler


def create_mlp_classifier_model(hidden_layer_sizes, activation, solver, alpha, max_iter, tol):
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
    )


def create_mlp_regressor_model(hidden_layer_sizes, activation, solver, alpha, max_iter, tol):
    return MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
    )


def train(model, x_train, x_test, y_train, y_test):
    scaler = StandardScaler()
    scaler.fit(x_train)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    model.fit(x_train, y_train)

    y_predict = model.predict(x_test)

    score = model.score(x_test, y_test)
    mape = mean_absolute_percentage_error(y_test, y_predict)

    print(f'\n{type(model).__name__}\nScore: {score}\nMAPE: {mape}')
