import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

def success(x):
    if (x>6.5):
        return 1
    else:
        return 0    


#convert date from inital format to year, month day category
netflix=pd.read_csv("NetflixOriginals.csv")
netflix["Premiere"] = pd.to_datetime(netflix['Premiere'])
netflix['Year'] = netflix['Premiere'].dt.year
netflix['Month'] = netflix['Premiere'].dt.month
netflix['Day'] = netflix['Premiere'].dt.day

le = preprocessing.LabelEncoder()
netflix["IMDB Score"]=netflix["IMDB Score"].apply(success)
netflix["IMDB Score"]=netflix["IMDB Score"].astype(int)
data1=netflix[['Genre', 'Runtime', 'Language','Year','IMDB Score']]
data1.to_csv('train/train.csv',index=False)
netflix["Language"]=le.fit_transform(netflix["Language"])
netflix["Genre"]=le.fit_transform(netflix["Genre"])
cols=['Genre', 'Runtime',  'Language','Year']
target=netflix['IMDB Score']
data=netflix[cols]
data_train, data_test, target_train, target_test = train_test_split(data,target, test_size = 0.30)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
#create object of the lassifier
neigh = KNeighborsClassifier(n_neighbors=3)
#Train the algorithm
neigh.fit(data_train, target_train)
# predict the response
import joblib
joblib.dump(neigh, 'model.pkl')
print("Model dumped!")

# Load the model that you just saved
neigh = joblib.load('model.pkl')

# Saving the data columns from training
model_columns = list(data_train.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")


