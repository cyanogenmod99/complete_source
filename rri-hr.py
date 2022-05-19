import pandas as pd
from scipy import signal
loadfile = 'rr.txt'
# 
df1 = pd.read_table(loadfile, header=None, sep=" " , names=['time','rr'])
df1.describe()

from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt

idx_nm_1 = df1[df1['rr'] >= 1.1].index
#df.drop('offline', axis=1)
df_test = df1.drop(idx_nm_1).reset_index(drop=True)

# x axis = time, y axis = rri
X = df_test.iloc[:,0].values
y = df_test.iloc[:,1].values
# 
x = np.arange(0,int(df_test.iloc[-1]['time']+1),1)
# cubic
cubic = interp1d(X,y, kind="cubic")
# 
plt.figure(figsize=(15, 5))
plt.plot(x,cubic(x),c="red",label="")
# 
print(x.shape)

df_beat = pd.DataFrame()
for i in range(len(cubic(x))):
        number = pd.Series([i,1/cubic(x[i])*60])
        df_beat = df_beat.append(number, ignore_index=True)

df_beat.columns = ['timestamp', 'HR']
df_beat.to_csv("rri to hr_drop.csv", index=False)