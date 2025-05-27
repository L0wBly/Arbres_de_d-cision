from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16]]
y = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # classe impair=1, pair=0



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=None)


clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy : {accuracy:.2f} ")


plt.figure(figsize=(12, 9))
plot_tree(clf, feature_names=["x"], class_names=["0", "1"], filled=True, fontsize=10, rounded=True)
plt.show()