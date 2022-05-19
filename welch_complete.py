import numpy as np
import pandas as pd
import scipy.interpolate as ip
from scipy.interpolate import splrep, splev
import matplotlib.pyplot as plt
import time
import scipy.signal as sig

now = str(int(time.time()))

df = pd.read_csv("physical-test-rr-recordings.csv")

idx_nm_1 = df[df['duration'] >= 1.0].index
df.drop('offline', axis=1)

df_test = df.drop(idx_nm_1)
df_test = df_test.reset_index(drop=True)

x = np.linspace(0, len(df_test['duration'])-1, len(df_test['duration'])) # X axis => time
#x = 0~len(df_test['duration'])-1 sec

y = df_test['duration'].values # y axis => RR interval values

interval = 2*len(df_test['duration'])

spl = splrep(x,y) #spline 

x1 = np.linspace(0,len(df_test['duration'])-1, interval)
#interver 0.5s

y1 = splev(x1, spl)

s_time = x1.tolist()
s_rr = y1.tolist()

fs = 2 # 2Hz => Re-sampling rate

HF_list = list()
LF_HF_list = list()


#Welch's Method

for i in range(0, interval-40):
    x = s_rr[i:i+39]
    x_array = np.array(x)*1000
    freq, p1 = sig.welch(x_array, fs=fs, window='hanning', nperseg=len(x))

    print(freq)
    print(p1)

    vlf = 0.04 
    lf = 0.15
    hf = 0.4

    Fs = 500

    lf_freq_band = (freq >= vlf) & (freq <= lf)
    hf_freq_band = (freq >= lf) & (freq <= hf)

    dy = 1.0/ Fs

    LF = np.trapz(y=abs(p1[lf_freq_band]), x=None, dx=dy)
    HF = np.trapz(y=abs(p1[hf_freq_band]), x=None, dx=dy)
    LF_HF = float(LF) / HF

    HF_list.append(HF)
    LF_HF_list.append(LF_HF)

HF_df = pd.DataFrame(HF_list, columns=['HF'])
LF_HF_df = pd.DataFrame(LF_HF_list, columns=['LF/HF'])

#CSV

output_df = pd.concat([HF_df, LF_HF_df], axis=1)

output_df.to_csv("RRI_to_HFLF"+ now +".csv", index=False)

# Graph

fig, ax1 = plt.subplots()
ax1.set_xlabel('ts')

ax1.plot(HF_df.values, color='#ff7f00', label="HF")
ax1.plot(LF_HF_df.values, color='g', label="LF/HF")

plt.grid(True)
plt.legend(loc='upper right')
plt.savefig("HF_LFHF_graph"+ now +".png")

