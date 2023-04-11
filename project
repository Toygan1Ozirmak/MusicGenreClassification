import numpy as np
import os
import math
import operator
import pandas as pd
import seaborn as sns
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics



"""importing our libraries"""


cols = pd.read_csv('C:\musicml_fixed.csv', nrows=1).columns
df = pd.read_csv ('C:\musicml_fixed.csv', usecols=cols, error_bad_lines=False,low_memory=False)

"""reading our csv dataset with pandas library"""



print(df)
"""we can use .head or .tail"""


"""plt.scatter(df['instance_id'], df['popularity'])"""   """it is for visualize data"""


Dup_Rows = df[df.duplicated()]



print("\n\nDuplicate Rows : \n {}".format(Dup_Rows))


print("printing dropped duplicates \n")



nondup=df.iloc[10000:10005]
print(nondup)

df.drop([10000, 10001, 10002, 10003, 10004], inplace = True)

print("We gonna see the size \n",df.shape)

df.dropna(inplace=True) 
"""some rows are still have nan values we gonna clear that"""

df["key"].unique()
df["mode"].unique()
df["music_genre"].unique()

df = df.drop([ "instance_id", "track_name", "obtained_date"], axis = 1)
"""we are clearing this data because we dont need to these datas for prediction"""
print("we are looking at cleared data set \n", df.head())

print("We gonna see how we have empity field at artist name \n\n")
print(df[df["artist_name"] == "empty_field"])

df = df.drop(df[df["artist_name"] == "empty_field"].index)
top_10_artists = df["artist_name"].value_counts()[:10].sort_values(ascending = True)
""""print("\nTop 10 Artist List\n", top_10_artists)"""




plt.barh(top_10_artists.index, top_10_artists)
plt.xlabel("Number of songs per artist")
plt.title("Songs per artist")
plt.show()
"""sns.relplot(x="loudness", y="instrumentalness", data=df)"""
"""sns.relplot(x="danceability", y="tempo", data=df);"""
"""sns.boxplot(x='day', y='total_bill', data=df,)"""
"""sns.relplot(x="tempo", y="danceability", kind="line", data=df)"""


df.drop("artist_name", axis = 1, inplace = True) 
"""we are dropping artist list
 because artist list is not numeric so we are not gonna use
 for predictions"""

def plot_counts(feature, order = None):
            sns.countplot(x = feature, data = df, palette = "rocket", order = order)
            plt.title(f"Counts in each {feature}")
            plt.show()
plot_counts("key", ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"])
plot_counts("mode")
plt.figure(figsize = (20, 9))
plot_counts("music_genre")


df = df.drop(df[df["tempo"] == "?"].index)

"""Clearing our datas as tempo equals ?"""



df["tempo"] = df["tempo"].astype("float")
df["tempo"] = np.around(df["tempo"], decimals = 2)



numerics = df.drop(["key", "music_genre", "mode"], axis = 1)







encode = LabelEncoder()
df["key"] = encode.fit_transform(df["key"])

"""We transformed keys to the numerics like 1-2-3-4..."""
print(encode.classes_)


encodemode = LabelEncoder()
df["mode"] = encodemode.fit_transform(df["mode"])
print(df)
fig = plt.figure(figsize =(10, 7))
plt.boxplot(numerics["popularity"])


df = shuffle(df) 

"""shuffling main dataset for train set"""




music_features = df.drop(["music_genre"], axis = 1).values
music_genre = df["music_genre"].values





"""Creating test and training data"""

train_features, test_features, train_genre, test_genre = train_test_split(
      music_features, music_genre, random_state = 0, test_size = 0.50)


"""Doing Normalization our numeric values"""
scaler = StandardScaler()


music_features_scaled = np.array(scaler.fit_transform(music_features))
music_features_scaled.mean(), music_features_scaled.std()
scaler.fit(train_features)

train_features = scaler.transform(train_features)
test_features = scaler.transform(test_features)
        


"""knn = KNeighborsClassifier(n_neighbors = 1)
 
knn.fit(train_features, train_genre)
pred = knn.predict(train_features)
 
# Predictions and Evaluations
# Let's evaluate our KNN model !
from sklearn.metrics import classification_report, confusion_matrix


print(confusion_matrix(test_genre, pred))
 
print(classification_report(test_genre, pred))


error_rate = []
 
# Will take some time
for i in range(1, 40):
     
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(train_features, train_genre)
    pred_i = knn.predict(train_features)
    error_rate.append(np.mean(pred_i != train_genre))
 
plt.figure(figsize =(10, 6))
plt.plot(range(1, 40), error_rate, color ='blue',
                linestyle ='dashed', marker ='o',
         markerfacecolor ='red', markersize = 10)
 
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


knn = KNeighborsClassifier(n_neighbors = 1)
 
knn.fit(train_features, train_genre)
pred = knn.predict(train_features)
 
print('WITH K = 1')
print('\n')
print(confusion_matrix(train_genre, pred))
print('\n')
print(classification_report(train_genre, pred))
 
 
# NOW WITH K = 15
knn = KNeighborsClassifier(n_neighbors = 15)
 
knn.fit(train_features, train_genre)
pred = knn.predict(train_features)
 
print('WITH K = 15')
print('\n')
print(confusion_matrix(train_genre, pred))
print('\n')
print(classification_report(train_genre, pred))"""






from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier


model = OneVsOneClassifier(LogisticRegression())
model.fit(train_features, train_genre)




logisticRegr = LogisticRegression()
logisticRegr.fit(train_features, train_genre)
logisticRegr.predict(test_features[0].reshape(1,-1))

predictions = logisticRegr.predict(test_features)

score = logisticRegr.score(test_features, test_genre)
print("\n Acurracy of logistic reggresion",score)


cm = metrics.confusion_matrix(test_genre, predictions)
print("\n",cm)
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);



from sklearn import svm



clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets




clf.fit(train_features, train_genre)

#Predict the response for test dataset



y_pred = clf.predict(test_features)



print("\nAccuracy Of Support Vector Machine:", metrics.accuracy_score(test_genre, y_pred))
