import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import gzip

# Data loading
def load(path):
    # y = int labels
    # X = float32 images
    with gzip.open(path, "rt") as f:
        data = np.loadtxt(f, delimiter=",", skiprows=1)
    y = data[:, 0].astype(int)
    X = data[:, 1:] / 255.0 #Normalize
    return X.astype(np.float32), y

# Helper
def sigmoid(x):
    # σ(x) = 1 / (1 + e^{-x})
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))

def one_hot(labels, K):
    # turn integer labels into one-hot rows
    Y = np.zeros((labels.size, K), dtype=np.float32)
    Y[np.arange(labels.size), labels] = 1.0
    return Y

def accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()

# Model
class MLP:
    """
    A simple 3-layer neural net:
    input  -> hidden (sigmoid) -> output (sigmoid)
    """

    def __init__(self, n_inputs, n_hidden, n_outputs, seed=2802):
        rng = np.random.default_rng(seed)
        self.W1 = rng.uniform(-1.0, 1.0, size=(n_inputs + 1, n_hidden)).astype(np.float32)
        self.W2 = rng.uniform(-1.0, 1.0, size=(n_hidden + 1, n_outputs)).astype(np.float32)

    def forward(self, X):
        # B = batch size/number of samples
        B = X.shape[0]

        # Add bias to inputs and compute hidden layer
        X_bias = np.concatenate([X, np.ones((B, 1), dtype=X.dtype)], axis=1)
        hidden = X_bias @ self.W1
        h = sigmoid(hidden)

        # Add bias to hidden activations and compute outputs
        hidden_bias = np.concatenate([h, np.ones((B, 1), dtype=h.dtype)], axis=1)
        output = hidden_bias @ self.W2
        y = sigmoid(output)

        return X_bias, h, hidden_bias, y

    def predict(self, X):
        # Class prediction = index of the largest output neuron
        _, _, _, y = self.forward(X)
        return np.argmax(y, axis=1)

    def fit(self, X, Y_onehot, X_test, y_test, epochs=30, batch_size=20, eta=3.0, record_curve=True, verbose=False):
        # Verbose set to False to clean up output
        num_samples = X.shape[0]
        acc_history = []

        for epoch in range(1, epochs + 1):
            # Shuffles the data once per epoch to avoid bias
            order = np.random.permutation(num_samples)
            X_shuf = X[order]
            Y_shuf = Y_onehot[order]

            # Goes over the data in mini batches
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                X_batch = X_shuf[start:end]
                Y_batch = Y_shuf[start:end]
                B = end - start

                # Forward Propagation
                X_bias, h, hidden_bias, y_hat = self.forward(X_batch)

                # Back Propagation
                delta_out = y_hat * (1.0 - y_hat) * (Y_batch - y_hat)

                # Hidden layer delta
                W2_noBias = self.W2[:-1, :]
                delta_hid = (delta_out @ W2_noBias.T) * h * (1.0 - h)

                # Gradients
                grad_W2 = (hidden_bias.T @ delta_out) / B
                grad_W1 = (X_bias.T @ delta_hid) / B

                # Weight updates
                self.W2 += eta * grad_W2
                self.W1 += eta * grad_W1

            # Tracks test accuracy per epoch
            if record_curve:
                acc = accuracy(y_test, self.predict(X_test))
                acc_history.append(acc)
                if verbose:
                    print(f"Epoch {epoch:3d} | test acc = {acc*100:5.2f}%")

        return acc_history

def main():
    # default base args: 784 30 10 fashion-mnist_train.csv.gz fashion-mnist_test.csv.gz
    p = argparse.ArgumentParser()
    p.add_argument("NInput", type=int)
    p.add_argument("NHidden", type=int)
    p.add_argument("NOutput", type=int)
    p.add_argument("train_csv_gz")
    p.add_argument("test_csv_gz")
    args = p.parse_args()

    # Load data
    Xtr, ytr = load(args.train_csv_gz)
    Xte, yte = load(args.test_csv_gz)

    # One-hot labels for training
    Ytr = one_hot(ytr, args.NOutput)

    #Task 1 Max accuracy with default hyper parameters
    print("\nTask 1: Test Accuracy vs Epoch")
    mlp = MLP(args.NInput, args.NHidden, args.NOutput)
    acc_curve = mlp.fit(Xtr, Ytr, Xte, yte, epochs=30, batch_size=20, eta=3.0)

    plt.figure()
    plt.title("Task 1: Test Accuracy vs Epoch | Default hyperparameters", loc="left")
    plt.plot(np.arange(1, len(acc_curve)+1), np.array(acc_curve)*100, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Test accuracy (%)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("accuracy_vs_epoch.png", dpi=150)
    plt.close()
    print(f"With: epochs={30}, batch={20}, η={3.0} -> max acc = {max(acc_curve)*100:.2f}%")
    print("Plot saved as accuracy_vs_epoch.png.")
    print("\nTask completed 1/4.")

    # Task 2 Max accuracy with different learning rates(etas = [0.001, 0.01, 1.0, 10, 100])
    print("\nTask 2: Test accuracy vs Epoch with different learning rates(η)")
    etas = [0.001, 0.01, 1.0, 10, 100]
    plt.figure()
    plt.title("Task 2: Test Accuracy vs Epoch | Different learning rates(η)", loc="left")
    for eta in etas:
        mlp = MLP(args.NInput, args.NHidden, args.NOutput)
        acc_curve = mlp.fit(Xtr, Ytr, Xte, yte, epochs=30, batch_size=20, eta=eta)
        plt.plot(np.arange(1, len(acc_curve)+1), np.array(acc_curve)*100, marker="o", label=f"η={eta}")
        print(f"With: η = {eta:>5} -> max acc = {max(acc_curve)*100:.2f}%")
    plt.xlabel("Epoch")
    plt.ylabel("Test accuracy (%)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("accuracy_vs_epoch_different_learning_rates.png", dpi=150)
    plt.close()
    print("Plot saved as accuracy_vs_epoch_different_learning_rates.png.")
    print("\nTask completed 2/4.")

    # Task 3 Max accuracy with batch sizes = [1, 5, 20, 100, 300]
    print("\nTask 3: Max accuracy vs mini-batch sizes")
    batch_sizes = [1, 5, 20, 100, 300]
    max_by_batch = {}
    for b in batch_sizes:
        mlp = MLP(args.NInput, args.NHidden, args.NOutput)
        acc_curve = mlp.fit(Xtr, Ytr, Xte, yte, epochs=30, batch_size=b, eta=3.0)
        max_acc = max(acc_curve) * 100
        max_by_batch[b] = max_acc
        print(f"with: mini-batch = {b:>3}: -> max acc = {max_acc:.2f}%")
    plt.figure()
    plt.title("Task 3: Max Test Accuracy vs Mini-batch size | Default hyperparameters", loc="left")
    plt.plot(batch_sizes, [max_by_batch[b] for b in batch_sizes], marker="o")
    plt.xlabel("Mini-batch size")
    plt.ylabel("Max test accuracy (%)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("max_accuracy_vs_mini-batch.png", dpi=150)
    plt.close()
    print("Plot saved as max_accuracy_vs_mini-batch.png.")
    print("\nTask completed 3/4.")

    # Task 4: Best accuracy from different hyper-parameters
    print("\nTask 4: Best accuracy from different hyper-parameters")
    experiments = [
        # (epochs, batch, eta)
        (20, 40, 1.0),
        (30, 30, 2.0),
        (40, 20, 3.0),
    ]
    best = None
    plt.figure()
    for (ep, b, lr) in experiments:
        mlp = MLP(args.NInput, args.NHidden, args.NOutput)
        acc_curve = mlp.fit(Xtr, Ytr, Xte, yte, epochs=ep, batch_size=b, eta=lr)
        mx = max(acc_curve) * 100
        print(f"With: epochs={ep}, batch={b}, η={lr} -> max acc = {mx:.2f}%")
        if (best is None) or (mx > best[0]):
            best = (mx, (ep, b, lr))
        label = f"ep={ep}, b={b}, η={lr}"
        plt.plot(np.arange(1, len(acc_curve)+1), np.array(acc_curve)*100, marker="o", label=label)
    best_acc, (ep, b, lr) = best
    print(f"Highest accuracy: {best_acc:.2f}% with epochs={ep}, batch={b}, η={lr}")
    plt.xlabel("Epoch")
    plt.ylabel("Test accuracy (%)")
    plt.title("Task 4: Test Accuracy vs Epoch | Different hyperparameters", loc="left")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("accuracy_vs_epoch_hyperparameters.png", dpi=150)
    plt.close()
    print("Plot saved as accuracy_vs_epoch_hyperparameters.png.")
    print("\nTask completed 4/4.")
    print()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python nn.py NInput NHidden NOutput train.csv.gz test.csv.gz")
        sys.exit(1)
    main()

# Launch codes
# python nn.py 784 30 10 fashion-mnist_train.csv.gz fashion-mnist_test.csv.gz