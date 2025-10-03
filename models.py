from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def train_logistic_regression(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return classification_report(y_test, preds)

def train_decision_tree(X_train, X_test, y_train, y_test):
    # Removing max_depth and other restrictions to allow the model to learn more
    model = DecisionTreeClassifier(max_depth=10,min_samples_leaf=5,min_samples_split=10,random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return classification_report(y_test, preds)

def train_random_forest(X_train, X_test, y_train, y_test):
    # Increased n_estimators and removed restrictive parameters
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return classification_report(y_test, preds)