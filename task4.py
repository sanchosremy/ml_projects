import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

music_data = pd.read_csv('music.csv')
X = music_data.drop(columns=['genre'])
print(X)

Y = music_data['genre']
print(Y)

model = DecisionTreeClassifier()
model.fit(X, Y)

tree.export_graphviz(model, out_file='music-recommender.dot',
                     feature_names=['age', 'gender'],
                     class_names=sorted(Y.unique()),
                     label='all',
                     rounded=True,
                     filled=True)