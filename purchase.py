import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
x=pd.read_excel(r"C:\Users\akshi\Downloads\Lab Session1 Data.xlsx",sheet_name='Purchase data')
x.drop(x.iloc[:,5:22],inplace=True,axis=1)
A=x.iloc[:,1:-1].values
C=x.iloc[:,-1].values
A=np.array(A)
C=np.array(C)
print("Matrix of A:")
print(A)
print("Matrix of C:")
print(C)
rank=np.linalg.matrix_rank(A)
print("Rank of Matrix A:", rank)
inverse=np.linalg.pinv(A)
print("Inverse of A: ",inverse)
Pseudo_inv=np.matmul(inverse,C)
print("Pseudo inverse is ie actual cost of each product is : ",Pseudo_inv)
y=np.array(x['Payment (Rs)'])
num=len(y)
f=[]
for i in range(0,num):
    if y[i]>200:
        f.append('RICH')
    else:
        f.append('POOR')
x.insert(loc = 5,column = 'Label',value = f)
print("New Data Excel Sheet for Purchase Data is: ")
print(x)
n = x.drop(['Customer', 'Payment (Rs)', 'Label'], axis=1) 
m = x['Label']
n_train, n_test, m_train, m_test = train_test_split(n, m, test_size=0.2, random_state=42) 
scaler = StandardScaler()
n_train_scaled = scaler.fit_transform(n_train)
n_test_scaled = scaler.transform(n_test)
model = RandomForestClassifier(random_state=42)
model.fit(n_train_scaled, m_train)
m_pred = model.predict(n_test_scaled) 
print(classification_report(m_test, m_pred))
