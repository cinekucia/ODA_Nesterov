from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


def load_diabetes_data(test_size=0.2, random_state=42):
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test
