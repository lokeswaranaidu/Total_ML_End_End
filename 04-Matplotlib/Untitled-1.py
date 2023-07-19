

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec

data = pd.read_csv(r"C:\Users\User\Desktop\UDEMY  FOLDER\archive\creditcard.csv")
data.head()

data.shape
data.describe()


fraud=data[data['Class']==1]
valid=data[data['Class']==0]

outlierFraction=len(fraud)/float(len(valid))
print(outlierFraction)

print('Fraud Cases: {}'.format(len(data[data['Class'] == 1])))
print('Valid Transactions: {}'.format(len(data[data['Class'] == 0])))

fraud.Amount.describe()

print(data.shape)
print(data.describe)

corrmat=data.corr()

fig=plt.figure(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.8, square=True)
plt.show()

X=data.drop(['Class'], axis=1)
Y=data['Class']
print(X.shape)
print(Y.shape)

xData=X.values
yData=Y.values