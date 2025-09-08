import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score
import math

def read_positive_int(prompt: str) -> int:
    # Enter a positive integer number.
    while True:
        raw = input(prompt)
        try:
            n = int(raw.strip())
            if n > 0:
                return n
            print("Please enter a positive integer.")
        except ValueError:
            print("Please enter a valid integer (a positive integer).")

def read_nonneg_int(prompt: str) -> int:
    # Enter a 0 or positive integer number.
    while True:
        raw = input(prompt)
        try:
            n = int(raw.strip())
            if n >= 0:
                return n
            print("Please enter 0 or a positive integer.")
        except ValueError:
            print("Please enter a valid integer (0 or a positive integer).")

def _is_valid_float(x: float) -> bool:
    return not (math.isnan(x) or math.isinf(x))

def read_float(prompt: str) -> float:
    # Read real numbers
    while True:
        raw = input(prompt)
        try:
            v = float(raw.strip())
            if _is_valid_float(v):
                return v
            print("Invalid value (NaN/Inf).")
        except ValueError:
            print("Please enter a real number.")



def main():
    # Read the training set TrainS of size N
    N = read_positive_int("Please enter a positive integer number of training samples N: ")

    # Use NumPy to preconfigure (N,2) to store (x and y)
    train_data = np.empty((N, 2), dtype=float)

    # Enter x and y
    for i in range(N):
        x_i = read_float(f"Training {i+1}th x: ")
        y_i = read_nonneg_int(f"Training {i+1}th y: ")
        train_data[i, 0] = x_i
        train_data[i, 1] = y_i

    # Read the test set TestS of size M
    M = read_positive_int("Please enter the number of test samples M: ")

    # Use NumPy preconfigured (M,2) to store (x, y)
    test_data = np.empty((M, 2), dtype=float)

    # Enter x and y
    for i in range(M):
        x_i = read_float(f"Test {i+1}th x: ")
        y_i = read_nonneg_int(f"Test {i+1}th y: ")
        test_data[i, 0] = x_i
        test_data[i, 1] = y_i

    # Split into x / y required by sklearn
    X_train = train_data[:, 0].reshape(-1, 1)
    y_train = train_data[:, 1].astype(int)
    X_test  = test_data[:, 0].reshape(-1, 1)
    y_test  = test_data[:, 1].astype(int)

    # K range
    max_k = min(10, X_train.shape[0])
    param_grid = {"n_neighbors": list(range(1, max_k + 1))}
    clf = KNeighborsClassifier(weights="uniform", metric="minkowski")

    # CV: Prefer StratifiedKFold; fall back to KFold if some class counts < 2
    unique, counts = np.unique(y_train, return_counts=True)
    min_class = counts.min() if counts.size > 0 else 0
    desired_max_splits = min(5, X_train.shape[0]) if X_train.shape[0] > 1 else 2

    if min_class >= 2:
        n_splits = max(2, min(desired_max_splits, int(min_class)))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        n_splits = max(2, min(desired_max_splits, X_train.shape[0]))
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        cv=cv,
        n_jobs=-1,
        scoring="accuracy",
        refit=True
    )
    grid.fit(X_train, y_train)

    # TestS predict and  evaluate
    best_k = int(grid.best_params_["n_neighbors"])
    y_pred = grid.best_estimator_.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    # Result
    print("Best k:", best_k)
    print("Test Accuracy:", round(float(test_acc), 6))

if __name__ == "__main__":
    main()