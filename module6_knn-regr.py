import numpy as np

class KNNRefressor1D:
    def __init__(self):
        # shape = (n_samples, 2)
        self.data = np.empty((0, 2), dtype=float)

    def insert_point(self, x: float, y: float) -> None:
        self.data = np.vstack([self.data, np.array([[x, y]], dtype=float)])

    def predict(self, x_query: float, k: int) -> float:
        n = self.data.shape[0]

        if k <= 0:
            raise ValueError("k must be the positive integer!!")
        
        if k > n:
            raise ValueError(f"k can not bigger than N which current N={n}, k={k})")
        
        # distance -> vcetor
        distances = np.abs(self.data[:, 0] - x_query)

        # Take the k indices with the smallest distance
        nn_idx = np.argsort(distances)[:k]

        #Calculate the average y of the k the nearest neighbors
        y_vals = self.data[nn_idx, 1]
        return float(np.mean(y_vals))
                     

def read_positive_int(prompt: str) -> int:
    while True:
        raw = input(prompt)
        try:
            n = int(raw)
            if n > 0:
                return n
            print("Please Enter a positive integer N: ")
        except ValueError:
            print("Please Enter a positive integer!!")


def read_float(prompt: str) -> float:
    while True:
        raw = input(prompt)
        try:
            return float(raw)
        except ValueError:
            print("Please enter a real number!!")


def main():
    knn = KNNRefressor1D()
    N = read_positive_int("Please enter a positive integer N: ")
    k = read_positive_int("Please enter a positive integer k: ")

    for i in range(1, N + 1):
        x_i = read_float(f"{i} position's x is: ")
        y_i = read_float(f"{i} position's y is: ")
        knn.insert_point(x_i, y_i)

    X = read_float("Please enter the number you want to search X: ")

    try:
        y_hat = knn.predict(X, k)
        print(y_hat)
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
