import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import tree, metrics
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# Partie Iris
iris = pd.read_csv('Iris.csv')

X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris['Species']

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.30, random_state=101 )
clf = DecisionTreeClassifier(criterion='gini')
clf.fit(X_train, y_train)
print(clf.get_depth())

plt.figure(figsize=(10, 8))
tree.plot_tree(clf,rounded=True, filled=True)
plt.show()

y_pred = clf.predict(X_test)

print(metrics.classification_report(y_test, y_pred))
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels= ['Setosa', 'Versicolor', 'Virginica'])


clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train, y_train)
print(clf.get_depth())

plt.figure(figsize=(10, 8))
tree.plot_tree(clf,rounded=True, filled=True)
plt.show()

y_pred = clf.predict(X_test)

print(metrics.classification_report(y_test, y_pred))
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels= ['Setosa', 'Versicolor', 'Virginica'])

# Partie COVID
covid = pd.read_csv('country_vaccinations.csv')
covid = covid.dropna()

le_date = LabelEncoder()
le_vaccines = LabelEncoder()
le_country = LabelEncoder()

covid['date_encoded'] = le_date.fit_transform(covid['date'])
covid['vaccines_encoded'] = le_vaccines.fit_transform(covid['vaccines'])
covid['country_encoded'] = le_country.fit_transform(covid['country'])

X = covid[['date_encoded', 'vaccines_encoded', 'daily_vaccinations_per_million']]
y = covid['country_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

clf = DecisionTreeClassifier(criterion='gini')
clf.fit(X_train, y_train)

print("Profondeur de l’arbre :", clf.get_depth())

def custom_plot_tree(model, feature_names, class_names):
    plt.figure(figsize=(18, 10))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        impurity=False,
        proportion=False,
        label='none',  # Supprime samples / value
        fontsize=10,
    )
    plt.title("Arbre de Décision : Prédiction des pays", fontsize=16)
    plt.show()

y_pred = clf.predict(X_test)

# --- VERSION LISIBLE : top 10 pays les plus fréquents dans y_test ---

# Trouver les 10 classes les plus fréquentes dans y_test
counter = Counter(y_test)
top10_labels = [label for label, count in counter.most_common(20)]
top10_class_names = le_country.inverse_transform(top10_labels)

# Filtrer y_test et y_pred pour ne garder que ces classes
mask_test = np.isin(y_test, top10_labels)
y_test_top10 = y_test[mask_test]
y_pred_top10 = y_pred[mask_test]

print(metrics.classification_report(y_test_top10, y_pred_top10,
                                    labels=top10_labels,
                                    target_names=top10_class_names))

confusion_matrix = metrics.confusion_matrix(y_test_top10, y_pred_top10, labels=top10_labels)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                            display_labels=top10_class_names)
cm_display.plot(xticks_rotation=45)
plt.show()