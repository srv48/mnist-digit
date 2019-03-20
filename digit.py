import numpy as np
import pandas as pd

dataset=pd.read_csv('train.csv')
y=dataset.iloc[:,0].values
X=dataset.iloc[:, 1:].values

test = pd.read_csv('test.csv')
X_test = test.iloc[:, :].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X = sc.fit_transform(X)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
rfc.fit(X, y)

pred = rfc.predict(X_test)

ind = [i+1 for i in range(28000) ]
df_out = pd.DataFrame({'ImageId':ind, 'Label':pred})
df_out.to_csv("out.csv", index = False)