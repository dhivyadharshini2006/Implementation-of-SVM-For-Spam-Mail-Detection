# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Detect the encoding of the `spam.csv` file and load it using the detected encoding.
2. Check basic data information and identify any null values.
3. Define the features (`X`) and target (`Y`), using `v2` as the feature (message text) and `v1` as the target (spam/ham label).
4. Split the data into training and testing sets (80-20 split).
5. Use `CountVectorizer` to convert the text data in `X` to a matrix of token counts, fitting on the training set and transforming both training and test sets.
6. Initialize and train an SVM classifier on the transformed training data.
7. Predict the target labels for the test set.
8. Calculate and display the model's accuracy.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by:Dhivya Dharshini B
RegisterNumber:  212223240031
*/

```


```
import chardet
file='spam.csv'
with open(file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result
import pandas as pd 
data=pd.read_csv("spam.csv",encoding="Windows-1252")
data.head()
data.info()
data.isnull().sum()
x=data["v2"].values
y=data["v1"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:
### OUTPUT 1: ![image](https://github.com/user-attachments/assets/a737d693-b178-40b9-b3b4-c9a1d2d04cb9)

### OUTPUT 2:  ![image](https://github.com/user-attachments/assets/ed3eb8c6-3c73-47bf-9157-8ad687e38329)

### OUTPUT 3:  ![image](https://github.com/user-attachments/assets/1b5140d2-cf52-4545-8aad-a34188031356)

### OUTPUT 4:   ![image](https://github.com/user-attachments/assets/4775af96-c825-4c5c-b075-b0703f37ed4f)

### OUTPUT 5:   ![image](https://github.com/user-attachments/assets/14432860-613a-4c0b-a8cd-442c28d446bf)

### OUTPUT 6:   ![image](https://github.com/user-attachments/assets/60bec89e-c266-4fe8-b9de-96e621b94ec7)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
