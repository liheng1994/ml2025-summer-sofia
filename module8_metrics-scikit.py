import numpy as np
from sklearn.metrics import precision_score, recall_score

def read_positive_int(prompt: str) -> int:
    # Enter a positive integer number.
    while True:
        raw = input(prompt)
        try:
            n = int(raw.strip())
            if n > 0:
                return n
            print("Please enter an positive integer.")
        except ValueError:
            print("Please enter a valid integer(an positive integer).")


def read_binary(prompt: str) -> int:
    # Read binary tag 0 or 1.
    while True:
        raw = input(prompt)
        try:
            v = int(raw.strip())
            if v in (0, 1):
                return v
            print("Please enter 0 or 1.")
        except ValueError:
            print("Please enter 0 or 1.")


def main():
    # Read a positive integer number
    N = read_positive_int("Enter N (number of (x,y) pairs): ")

    # Use NumPy to preconfigure (N,2) to store (x=ground truth, y=pred)
    data = np.empty((N, 2), dtype=int)

    # Input: first x (ground truth), then y (prediction), both are 0/1
    for i in range(N):
        x_i = read_binary(f"[{i+1}] x (ground truth, 0 or 1): ")
        y_i = read_binary(f"[{i+1}] y (predicted, 0 or 1): ")
        data[i, 0] = x_i
        data[i, 1] = y_i

    # Read y_true and y_pred
    y_true = data[:, 0]
    y_pred = data[:, 1]

    # Calculating Precision/Recall with sklearn
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)

    
    print("Precision:", round(float(precision), 2))
    print("Recall   :", round(float(recall), 2))

if __name__ == "__main__":
    main()