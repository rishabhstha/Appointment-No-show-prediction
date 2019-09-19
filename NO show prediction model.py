import pylab
import calendar
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from datetime import datetime
import matplotlib.pyplot as plt

data=pd.read_csv("Dataset.csv")

#Just to remove confusions about spelling errors in further analysis
data.rename(columns = {'Hipertension':'Hypertension',
                         'Handcap': 'Handicap'}, inplace = True)

#Checking the unique values of the data set
print('Age:',sorted(data.Age.unique()))
print('Gender:',data.Gender.unique())
print("Neighbourhood",data.Neighbourhood.unique())
print('Scholarship:',data.Scholarship.unique())
print('Hypertension:',data.Hypertension.unique())
print('Diabetes:',data.Diabetes.unique())
print('Alcoholism:',data.Alcoholism.unique())
print('Handicap:',data.Handicap.unique())
print('SMS_received:',data.SMS_received.unique())
#print('No-show:',data.No-show.unique())


#Transforming the data into numbers
from sklearn import preprocessing
#Label Encoding of Gender, Neighbourhood, No-show 
le = preprocessing.LabelEncoder()

le.fit(data["Gender"])
data["Gender"]=le.transform(data["Gender"])


le.fit(data["No-show"])
data["No-show"]=le.transform(data["No-show"])

le.fit(data["Neighbourhood"])
data["Neighbourhood"]=le.transform(data["Neighbourhood"])
data.head()

le.fit(data["No-show"])
data["No-show"]=le.transform(data["No-show"])

#appointment date
data["AppointmentDay"] = data.AppointmentDay.apply(lambda x : x.split("T")[0])
data["AppointmentDay"].head(2)

#schedule date
data["ScheduledDay"] = data.ScheduledDay.apply(lambda x : x.split("T")[0])

import datetime
data.ScheduledDay = data.ScheduledDay.apply(np.datetime64)
data.AppointmentDay = data.AppointmentDay.apply(np.datetime64)


data['timeD']= (data.AppointmentDay -  data.ScheduledDay)/np.timedelta64(1, 'D')
print(data.ScheduledDay.head())
print(data.AppointmentDay.head())
print(data.timeD.head())
print(data.columns)
#print(noShow.AwaitingTime.head())

data.drop(data[data.timeD < 0].index, inplace=True)
data.drop(data[data.Age < 0].index, inplace=True)
data=data.drop(["AppointmentDay","ScheduledDay","PatientId","AppointmentID"],axis=1)
data.reset_index()
data.index.name="Index"
labels=data.pop("No-show")
labels.shape



from imblearn.over_sampling import SMOTE

df_new,lab_new=SMOTE(random_state=3).fit_sample(data,labels)
df_new=pd.DataFrame(df_new)
lab_new=pd.DataFrame(lab_new)

import numpy as np
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_new, lab_new, test_size=0.30, random_state=42)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

nn=MLPClassifier(activation='relu', solver='adam',hidden_layer_sizes=(100,), random_state=1,max_iter=200)
nn.fit(X_train, y_train)
print ("Accuracy for NN= ", accuracy_score(y_test, nn.predict(X_test)))