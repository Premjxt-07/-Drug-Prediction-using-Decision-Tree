# DEPENDENCIES

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.externals.six import StringIO
import matplotlib.pyplot as plt
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree
"%matplotlib inline"

df = pd.read_csv("drug200.csv", delimiter=",")
# print(df.head())

X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
# print(X)
y = df['Drug']


""" 
    Convert Catergorical object type data into numerical data using sklearn preprocessing fit and transform method.
"""

le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F', 'M'])
X[:, 1] = le_sex.transform(X[:, 1])

le_BP = preprocessing.LabelEncoder()
le_BP.fit(['LOW', 'NORMAL', 'HIGH'])
X[:, 2] = le_BP.transform(X[:, 2])

le_ch = preprocessing.LabelEncoder()
le_ch.fit(['NORMAL', 'HIGH'])
X[:, 3] = le_ch.transform(X[:, 3])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=3)


drugTree = DecisionTreeClassifier(criterion='entropy', max_depth=4)
drugTree.fit(X_train, y_train)
predTree = drugTree.predict(X_test)

print("Accuracy of Decision Tree is :",
      metrics.accuracy_score(y_test, predTree))


dot_data = StringIO()
filename = "drugtree.png"
featureNames = df.columns[0:5]
targetNames = df["Drug"].unique().tolist()
out = tree.export_graphviz(drugTree, feature_names=featureNames, out_file=dot_data, class_names=np.unique(
    y_train), filled=True,  special_characters=True, rotate=False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img, interpolation='nearest')
