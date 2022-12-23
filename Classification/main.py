from data import load_data
from models import create_decision_tree_model, create_random_forest_tree_model, train
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    X, y = load_data('../Exchange rates.csv')

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print('\n', X.head())

    decisionTreeModel = create_decision_tree_model(
        criterion='gini',
        splitter='best',
    )
    train(decisionTreeModel, X_train, X_test, Y_train, Y_test)

    randomForestTreeModel = create_random_forest_tree_model(
        n_estimators=300,
        criterion='gini',
        max_features='sqrt',
        bootstrap=True,
        n_jobs=-1,
    )
    train(randomForestTreeModel, X_train, X_test, Y_train, Y_test)
