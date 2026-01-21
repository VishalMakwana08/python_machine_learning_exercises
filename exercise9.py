from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Step 1: Load dataset
data = load_iris()
X = data.data      # features (input)
y = data.target    # labels (output)

print("Total data before split:")
print("X shape:", X.shape)
print("y shape:", y.shape)

# Step 2: Split dataset into Training and Test set
# test_size=0.20 means 20% data for testing, 80% for training
# random_state is used to get same output every time
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Step 3: Print results of split
print("\nAfter split:")
print("Training set X_train shape:", X_train.shape)
print("Training set y_train shape:", y_train.shape)

print("Test set X_test shape:", X_test.shape)
print("Test set y_test shape:", y_test.shape)

# Step 4: Show a small sample
print("\nFirst 3 training labels:", y_train[:3])
print("First 3 test labels:", y_test[:3])



