import pandas as pd
from sklearn.tree import DecisionTreeClassifier
#from sklearn.externals import joblib
import joblib

#music_data = pd.read_csv('music.csv')
#X = music_data.drop(columns=['genre'])
#print(X)

#Y = music_data['genre']
#print(Y)

#model = DecisionTreeClassifier()
#model.fit(X, Y)

model = joblib.load('music-recommender.joblib')
predictions = model.predict([[21, 1]])
print(predictions)
