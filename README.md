# EX 10 Implementation of SVM For Spam Mail Detection
## DATE:
## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Preprocessing
2. Feature Extraction
3. Model Training
4. Model Evaluation
5. Prediction

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection.
Developed by: Livya Dharshini G
RegisterNumber: 2305001013
*/
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split # Changed 'train_test_spilt' to 'train_test_split'
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score # Changed 'accuracy_scoore' to 'accuracy_score'
df=pd.read_csv('/content/spamEX10 ML.csv',encoding='ISO-8859-1')
df.head()
v=CountVectorizer()
X=v.fit_transform(df['v2'])
y=df['v1']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
model=svm.SVC(kernel='linear')
model.fit(X_train,y_train)
p=model.predict(X_test)
print("ACCURACY:",accuracy_score(y_test,p))
print("CLASSIFICATION REPORT:")
print(classification_report(y_test,p))
def predict_message(message):
  message_vec=v.transform([message])
  e=model.predict(message_vec)
  return e[0]
n="Congratulations!"
result=predict_message(n)
print(f"The message:'{n}' is classified as:{result}")
```

## Output:
![image](https://github.com/user-attachments/assets/b5cac753-c963-4723-b215-5d46af574321)
![image](https://github.com/user-attachments/assets/7f1bce93-6829-44c1-8e4d-b1398ed8a93b)
![image](https://github.com/user-attachments/assets/a7c58b0a-54a4-495d-af36-7e00e5bf9bd3)
![image](https://github.com/user-attachments/assets/7b0be1f2-c85a-4e7b-9641-e7f4031d769e)







## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
