from sklearn.model_selection import train_test_split

from models import create_mlp_classifier_model, create_mlp_regressor_model, train
from data import load_data

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    X, y = load_data('../Exchange rates.csv')

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlp_classifier = create_mlp_classifier_model(
        hidden_layer_sizes=(100, 100),
        activation='logistic',
        solver='lbfgs',
        alpha=1e-6,
        tol=1e-4,
        max_iter=2000,
    )

    mlp_regressor = create_mlp_regressor_model(
        hidden_layer_sizes=(100, 100),
        activation='logistic',
        solver='lbfgs',
        alpha=1e-6,
        tol=1e-4,
        max_iter=2000,
    )

    train(mlp_classifier, X_train, X_test, Y_train, Y_test)
    train(mlp_regressor, X_train, X_test, Y_train, Y_test)
