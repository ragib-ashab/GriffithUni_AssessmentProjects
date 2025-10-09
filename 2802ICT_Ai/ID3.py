import csv
import random
import math
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

# Data handling 
def read(path):
    # Opens CSV file and reads all rows
    with open(path, "r", newline="") as f:
        rows = list(csv.reader(f))

    # First row = header
    # Last column = label
    # Other rows = data
    header = rows[0]
    feature_names = header[:-1]
    data_rows = rows[1:]

    # x = features, list of dicts
    # y = class labels, list
    x, y = [], []
    for row in data_rows:
        if not row:
            continue
        # For each row make dict where key=feature name, value=feature value
        x.append({feature_names[i]: row[i] for i in range(len(feature_names))})
        y.append(row[-1])

    return feature_names, x, y

# Shuffles and split the dataset into train and test sets
def split_set(x, y, test_size=0.2, seed=123):
    rng = random.Random(seed)
    idxs = list(range(len(x)))
    rng.shuffle(idxs)

    # First "t" go to test, rest to train
    # basically makes a checkpoint at 20% of the data
    t = int(len(x) * test_size)
    test_idxs, train_idxs = idxs[:t], idxs[t:]

    return (
        [x[i] for i in train_idxs],
        [y[i] for i in train_idxs],
        [x[i] for i in test_idxs],
        [y[i] for i in test_idxs],
    )

#Helper
def mode(labels):
    # Returns the most common class label
    return Counter(labels).most_common(1)[0][0]


def entropy(labels):
    # Entropy measures "uncertainty" in labels
    total = len(labels)
    counts = Counter(labels)
    ent = 0.0
    for c in counts.values():
        p = c / total
        # entropy formula
        ent -= p * math.log2(p)
    return ent


def information_gain(x, y, feature):
    ent_before = entropy(y)

    # Group row indices by feature value
    groups = defaultdict(list)
    for i, row in enumerate(x):
        groups[row[feature]].append(i)

    # Computes remainder(A)
    remainder = 0
    total = len(y)
    for _, idxs in groups.items():
        subset = [y[j] for j in idxs]
        weight = len(idxs) / total
        remainder += weight * entropy(subset)

    # Information Gain = entropy before split - remainder(A)
    return ent_before - remainder


def best_feature(x, y, features):
    # Picks the feature with the highest information gain
    best, best_gain = None, -1.0
    for f in features:
        g = information_gain(x, y, f)
        if g > best_gain:
            best, best_gain = f, g
    return best


# Making the decision tree 
class Node:
    def __init__(self, feature=None, label=None):
        # feature = attribute to split on
        # label = final class label if leaf
        # children = feature value -> Child node
        self.feature = feature
        self.label = label
        self.children = {}

    def is_leaf(self):
        return self.label is not None


def make_tree(x, y, features_left):
    # If all labels are the same -> leaf
    if len(set(y)) == 1:
        return Node(label=y[0])

    # If no features left -> majority label
    if not features_left:
        return Node(label=mode(y))

    # Choose the best feature to split on
    best = best_feature(x, y, features_left)

    # If no useful split (gain ~ 0), return majority
    if information_gain(x, y, best) <= 1e-12:
        return Node(label=mode(y))

    # Creates decision node
    root = Node(feature=best)

    # Partition data by feature value
    groups = defaultdict(list)
    for i, row in enumerate(x):
        groups[row[best]].append(i)

    remain = [f for f in features_left if f != best]

    # Recursively build subtrees for each value
    for val, idxs in groups.items():
        xs = [x[i] for i in idxs]
        ys = [y[i] for i in idxs]
        root.children[val] = make_tree(xs, ys, remain)

    return root

# Prediction
def predict(root, row, default):
    node = root
    while not node.is_leaf():
        f = node.feature
        v = row.get(f, None)
        if v in node.children:
            node = node.children[v]
        else:
            # If value not seen in training, use majority label
            return default
    return node.label


def multi_predict(root, x, default):
    return [predict(root, r, default) for r in x]

# Evaluations
def compute_metrics(true, pred):
    labels = sorted(set(true) | set(pred))
    n = len(labels)
    lab_to_idx = {lab: i for i, lab in enumerate(labels)}

    # Builds confusion matrix
    cm = [[0] * n for _ in range(n)]
    for t, p in zip(true, pred):
        cm[lab_to_idx[t]][lab_to_idx[p]] += 1

    # Per-class precision, recall, f1
    precisions, recalls, f1s, support = {}, {}, {}, {}
    total = len(true)
    for i, lab in enumerate(labels):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(n)) - tp
        fn = sum(cm[i][c] for c in range(n)) - tp

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        precisions[lab] = prec
        recalls[lab] = rec
        f1s[lab] = f1
        support[lab] = sum(cm[i])

    # Overall accuracy
    acc = sum(cm[i][i] for i in range(n)) / total if total > 0 else 0.0

    # Macro averages = mean across classes
    macro_p = sum(precisions.values()) / n
    macro_r = sum(recalls.values()) / n
    macro_f1 = sum(f1s.values()) / n

    # Weighted averages = weighted by support
    ts = sum(support.values())
    weighted_p = sum(precisions[lab] * support[lab] for lab in labels) / ts
    weighted_r = sum(recalls[lab] * support[lab] for lab in labels) / ts
    weighted_f1 = sum(f1s[lab] * support[lab] for lab in labels) / ts

    return {
        "labels": labels,
        "confusion_matrix": cm,
        "accuracy": acc,
        "precision": precisions,
        "recall": recalls,
        "f1": f1s,
        "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
        "weighted": {"precision": weighted_p, "recall": weighted_r, "f1": weighted_f1},
        "support": support,
    }

def print_metrics(m):
    print("\nPer-class metrics:")
    print(f"{'Class':<12} {'Precision':>9} {'Recall':>9} {'F1':>9} {'Support':>9}")
    for lab in m["labels"]:
        print(f"{lab:<12} {m['precision'][lab]:9.3f} {m['recall'][lab]:9.3f} {m['f1'][lab]:9.3f} {m['support'][lab]:9d}")
    print("\nAverages:")
    print(f"  Macro      P={m['macro']['precision']:.3f} R={m['macro']['recall']:.3f} F1={m['macro']['f1']:.3f}")
    print(f"  Weighted   P={m['weighted']['precision']:.3f} R={m['weighted']['recall']:.3f} F1={m['weighted']['f1']:.3f}")

# Makes and Plot learning curve
def make_learning_curve(x_train, y_train, x_test, y_test, features, steps=10, seed=123):
    rng = random.Random(seed)
    idxs = list(range(len(x_train)))
    rng.shuffle(idxs)

    xs = [x_train[i] for i in idxs]
    ys = [y_train[i] for i in idxs]
    points = []

    for s in range(1, steps + 1):
        size = max(1, int(len(xs) * s / steps))
        x_small, y_small = xs[:size], ys[:size]

        # Train tree with smaller dataset slice
        root = make_tree(x_small, y_small, features[:])
        default = mode(y_small)

        # Test on full test set
        preds = multi_predict(root, x_test, default)
        acc = compute_metrics(y_test, preds)["accuracy"]
        points.append((size / len(xs), acc))

    return points


def plot_learning_curve(points, out_path):
    xs = [p[0] * 100 for p in points]
    ys = [p[1] * 100 for p in points]
    plt.figure()
    plt.plot(xs, ys, marker='o')
    plt.xlabel("Training data used (%)")
    plt.ylabel("Accuracy on test set (%)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    feat, x, y = read("car.csv")

    # (80/20) split for train/test
    x_train, y_train, x_test, y_test = split_set(x, y, test_size=0.2, seed=2802)

    # Build decision tree using training data
    tree = make_tree(x_train, y_train, feat[:])
    default = mode(y_train)

    # Predict on test data
    preds = multi_predict(tree, x_test, default)

    # Compute metrics
    metrics = compute_metrics(y_test, preds)
    print()
    print("Decision Tree (ID3) on car.csv")
    print(f"Training size: {len(x_train)}")
    print(f"Testing size : {len(x_test)}")
    print(f"Total accuracy: {metrics['accuracy']*100:.2f}%")
    print_metrics(metrics)
    print()

    # Make and save learning curve
    pts = make_learning_curve(x_train, y_train, x_test, y_test, feat, steps=10, seed=2802)
    print("Visual representation of learning curve saved as 'learning_curve.png'")
    plot_learning_curve(pts, "learning_curve.png")