from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def create_decision_tree_model(criterion, splitter):
    return DecisionTreeClassifier(
        criterion=criterion,
        splitter=splitter,
    )


def create_random_forest_tree_model(n_estimators, criterion, max_features, bootstrap, n_jobs):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        max_features=max_features,
        bootstrap=bootstrap,
        n_jobs=n_jobs,
    )


def train(model, x_train, x_test, y_train, y_test):
    scaler = StandardScaler()
    scaler.fit(x_train)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    model.fit(x_train, y_train)

    score = model.score(x_test, y_test)

    print(f'\nImportance of signs for {type(model).__name__}:', model.feature_importances_)
    print(f'Score: {score}')
