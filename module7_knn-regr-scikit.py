import math
import numpy as np
from sklearn.neighbors import KNeighborsRegressor


def read_positive_int(prompt: str) -> int:
    # Must enter an integer > 0. If enter an error, try again.
    while True:
        raw = input(prompt)
        try:
            n = int(raw.strip())
            if n > 0:
                return n
            print("Please enter an integer greater than 0. ")
        except ValueError:
            print("Please enter an positive integer. ")


def _is_valid_float(x: float) -> bool:
    return not (math.isnan(x) or math.isinf(x))


def read_float(prompt: str) -> float:
    while True:
        raw = input(prompt)
        try:
            v = float(raw.strip())
            if _is_valid_float(v):
                return v
            print("Invalid value (NaN/Inf). Please enter a real number. ")
        except ValueError:
            print("Please enter a real number. ")


def main():
    # Read N and k
    N = read_positive_int("Please enter a positive integer N: ")
    k = read_positive_int("Please enter a positive integer k: ")

    # Check if k is smaller than N
    if k > N:
        print(f"Error: k can not bigger than N (N={N}, k={k})! ")
        return
    
    # Use (N,2) to store (x,y)
    data = np.empty((N, 2), dtype= float)
    
    for i in range(N):
        x_i = read_float(f"{i+1} point's x: ")
        y_i = read_float(f"{i+1} point's y: ")
        data[i, 0] = x_i
        data[i, 1] = y_i

    # Search X
    Xq = read_float("Please enter X: ")
    X_train = data[:, 0].reshape(-1, 1)
    y_train = data[:, 1]

    # Build the modal. 
    if k <= 0:
        print("Error: k must be a positive integer. ")
        return
    
    try:
        model = KNeighborsRegressor(n_neighbors=k, weights="uniform")
        model.fit(X_train, y_train)
        y_hat = model.predict(np.array([[Xq]], dtype=float))[0]
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Calculate the number of variance for training y
    try:
        y_variance = float(np.var(y_train, ddof=0))
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Print result
    print("k-NN Regression (Y): ", round(float(y_hat), 6))
    print("y variance: ", round(y_variance, 6))


if __name__ == "__main__":
    main()