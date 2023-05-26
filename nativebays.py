from sklearn.datasets import fetch_openml
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# 1. Download data set from tensorflow
mnist = fetch_openml('mnist_784', version=1, parser='auto')

# 2. Check how the data looks, how many samples you have, what is the structure of the data. Save the information in log file.
with open('log.txt', 'w') as f:
    f.write(f"Number of samples: {len(mnist.data)}\n")
    f.write(f"Data structure: {mnist.data.shape}\n")

# 3. You can access images and labels as .data and .target
X = mnist.data.to_numpy()
y = mnist.target

# 4. Show few sample handwritten digits with label description in the title
fig, axes = plt.subplots(2, 5)
for i, ax in enumerate(axes.flat):
    ax.imshow(np.reshape(X[i], (28, 28)), cmap='gray')
    ax.set_title(f"Label: {y[i]}")
plt.show()

# 5. Split the data into training and test
X_train, X_test, y_train, y_test = train_test_split(X, y)

# 6. Think which version of naive bayes you should use and train the model using sklearn
clf = GaussianNB()
clf.fit(X_train, y_train)

# 7. Show confusion matrix in graphical way (use matplotlib or similar), describe why results looks like that
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, cmap='Blues')
plt.colorbar()
plt.show()

# 8. Show accuracy for whole test and error rate for each class. Think about other useful metrics.
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc}")