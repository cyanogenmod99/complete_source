import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd
import sys
import time


plt.rcParams['figure.figsize'] = 20, 8

#df = pd.read_csv("test_csv1623259598.csv")
#df_picked = df.loc[:, ["cnt"]]
# df_picked = df.loc[:, ["LF/HF"]]

df_arm = pd.read_csv('accel3_arm_mean_5.csv')
#df_neck = pd.read_csv('sensor_csv_mitsuke_neck_mean.csv')
#df_waist = pd.read_csv('sensor_csv_kobayashi_waist_mean.csv')

df_picked = df_arm.loc[:, ['x_mean']]
df_picked_y = df_arm.loc[:, ['y_mean']]
df_picked_z = df_arm.loc[:, ['z_mean']]

'''
df_p_neck_x = df_neck.loc[:, ['x_mean']]
df_p_neck_y = df_neck.loc[:, ['y_mean']]
df_p_neck_z = df_neck.loc[:, ['z_mean']]

df_p_waist_x = df_waist.loc[:, ['x_mean']]
df_p_waist_y = df_waist.loc[:, ['y_mean']]
df_p_waist_z = df_waist.loc[:, ['z_mean']]
'''

'''
df_picked = df[df['sensor_id'] == '810DA3B9'].loc[:, ["x"]]
df_picked.reset_index(drop=True, inplace=True)

df_picked_y = df[df['sensor_id'] == '810DA3B9'].loc[:, ["y"]]
df_picked_y.reset_index(drop=True, inplace=True)

df_picked_z = df[df['sensor_id'] == '810DA3B9'].loc[:, ["z"]]
df_picked_z.reset_index(drop=True, inplace=True)
'''

#print(type(df_picked))

#Norm calculate
norm = np.sqrt(df_arm['x_mean'].values * df_arm['x_mean'].values + df_arm['y_mean'].values * df_arm['y_mean'].values + df_arm['z_mean'].values * df_arm['z_mean'].values)
df_picked_norm = pd.DataFrame(data=norm, columns = ['norm'], dtype='float')
'''
norm_neck = np.sqrt(df_neck['x_mean'].values * df_neck['x_mean'].values + df_neck['y_mean'].values * df_neck['y_mean'].values + df_neck['z_mean'].values * df_neck['z_mean'].values)
df_picked_norm_neck = pd.DataFrame(data=norm, columns = ['norm'], dtype='float')

norm_waist = np.sqrt(df_waist['x_mean'].values * df_waist['x_mean'].values + df_waist['y_mean'].values * df_waist['y_mean'].values + df_waist['z_mean'].values * df_waist['z_mean'].values)
df_picked_norm_waist  = pd.DataFrame(data=norm, columns = ['norm'], dtype='float')
'''

#df_abnor_check = pd.read_csv("abnormal_cnt_byR.csv")
#df_abnor_check = df_abnor_check.iloc[:, 1]

w = 48
m = 2
k = 10
L = 16
Tt = len(df_picked)

abnor_score = np.zeros(Tt)
abnor_score_y = np.zeros(Tt)
abnor_score_z = np.zeros(Tt)
abnor_score_norm = np.zeros(Tt)

print(df_picked.head())
print(Tt)
program_start = time.time()
now = str(int(program_start))

def embed(df, size):
    window = pd.DataFrame()
    window=window.reset_index()
    for i in range(0, len(df)-size+1):
        tmp = df.iloc[i:i+size]
        tep = tmp.dropna()
        tmp = tmp.reset_index()
        window = pd.concat([window, tmp], axis=1)
    window = window.drop(columns="index")
    # window.to_csv("CSV_files/window.csv", index=False)

    return window

for t in range(w+k, Tt-L+1):
    tstart = t-w-k+1
    tend = t
    X1 = pd.DataFrame(embed(df_picked.iloc[tstart:tend, -1], w))
    X1 = X1.iloc[::-1]

    tstart = tstart + L
    tend = tend + L
    X2 = pd.DataFrame(embed(df_picked.iloc[tstart:tend, -1], w))
    X2 = X2.iloc[::-1]

    U1, s1, V1 = np.linalg.svd(X1.astype(np.float64), full_matrices=False)
    U1 = U1[:, 0:m]
    U2, s2, V2 = np.linalg.svd(X2.astype(np.float64), full_matrices=False)
    U2 = U2[:, 0:m]

    U3, s3, V3 = np.linalg.svd(np.dot(U1.T, U2))

    sig1 = s3[0]
    abnor_score[t] = 1 - (sig1 * sig1)
    #print(abnor_score[t])

avg_x = np.mean(abnor_score)
std_x = np.std(abnor_score)
th_x = avg_x + 0.5*std_x

change_x = abnor_score[abnor_score >= th_x]
change_x_df = pd.DataFrame(change_x, columns=['change point_x_arm'])

print(avg_x)
print(std_x)
print(th_x)
print(change_x)

print(t)
print("...X1...")
print(X1.head())
print("...X2...")
print(X2.head())
X1.to_csv("CSV_files/X1"+ now +".csv")
X2.to_csv("CSV_files/X2"+ now +".csv")
abnor_score_df = pd.DataFrame(abnor_score, columns=['abnormal_x_arm'])
data_and_abnor = pd.concat([df_arm.loc[:, ['x_mean']], abnor_score_df], axis=1)
#data_and_abnor = pd.concat([df.loc[:, ['instant', 'cnt']], abnor_score_df, df_abnor_check], axis=1)



for t in range(w+k, Tt-L+1):
    tstart = t-w-k+1
    tend = t
    X1_y = pd.DataFrame(embed(df_picked_y.iloc[tstart:tend, -1], w))
    X1_y = X1_y.iloc[::-1]

    tstart = tstart + L
    tend = tend + L
    X2_y = pd.DataFrame(embed(df_picked_y.iloc[tstart:tend, -1], w))
    X2_y = X2_y.iloc[::-1]

    U1_y, s1_y, V1_y = np.linalg.svd(X1_y.astype(np.float64), full_matrices=False)
    U1_y = U1_y[:, 0:m]
    U2_y, s2_y, V2_y = np.linalg.svd(X2_y.astype(np.float64), full_matrices=False)
    U2_y = U2_y[:, 0:m]

    U3_y, s3_y, V3_y = np.linalg.svd(np.dot(U1_y.T, U2_y))

    sig1_y = s3_y[0]
    abnor_score_y[t] = 1 - (sig1_y * sig1_y)

avg_y = np.mean(abnor_score_y)
std_y = np.std(abnor_score_y)
th_y = avg_y + 0.5*std_y

change_y = abnor_score_y[abnor_score_y >= th_y]
change_y_df = pd.DataFrame(change_y, columns=['change point_y_arm'])

print(t)
print("...X1...")
print(X1_y.head())
print("...X2...")
print(X2_y.head())
X1_y.to_csv("CSV_files/X1_y"+ now +".csv")
X2_y.to_csv("CSV_files/X2_y"+ now +".csv")
abnor_score_df_y = pd.DataFrame(abnor_score_y, columns=['abnormal_y_arm'])
data_and_abnor_y = pd.concat([df_arm.loc[:, ['y_mean']], abnor_score_df_y], axis=1)
#data_and_abnor = pd.concat([df.loc[:, ['instant', 'cnt']], abnor_score_df, df_abnor_check], axis=1)

for t in range(w+k, Tt-L+1):
    tstart = t-w-k+1
    tend = t
    X1_z = pd.DataFrame(embed(df_picked_z.iloc[tstart:tend, -1], w))
    X1_z = X1_z.iloc[::-1]

    tstart = tstart + L
    tend = tend + L
    X2_z = pd.DataFrame(embed(df_picked_z.iloc[tstart:tend, -1], w))
    X2_z = X2_z.iloc[::-1]

    U1_z, s1_z, V1_z = np.linalg.svd(X1_z.astype(np.float64), full_matrices=False)
    U1_z = U1_z[:, 0:m]
    U2_z, s2_z, V2_z = np.linalg.svd(X2_z.astype(np.float64), full_matrices=False)
    U2_z = U2_z[:, 0:m]

    U3_z, s3_z, V3_z = np.linalg.svd(np.dot(U1_z.T, U2_z))

    sig1_z = s3_z[0]
    abnor_score_z[t] = 1 - (sig1_z * sig1_z)

avg_z = np.mean(abnor_score_z)
std_z = np.std(abnor_score_z)
th_z = avg_z + 0.5*std_z

change_z = abnor_score_z[abnor_score_z >= th_z]
change_z_df = pd.DataFrame(change_z, columns=['change point_z_arm'])


print(t)
print("...X1_z...")
print(X1_z.head())
print("...X2_z...")
print(X2_z.head())
X1_z.to_csv("CSV_files/X1_z"+ now +".csv")
X2_z.to_csv("CSV_files/X2_z"+ now +".csv")
abnor_score_df_z = pd.DataFrame(abnor_score_z, columns=['abnormal_z_arm'])
data_and_abnor_z = pd.concat([df_arm.loc[:, ['z_mean']], abnor_score_df_z], axis=1)
#data_and_abnor = pd.concat([df.loc[:, ['instant', 'cnt']], abnor_score_df, df_abnor_check], axis=1)

for t in range(w+k, Tt-L+1):
    tstart = t-w-k+1
    tend = t
    X1_norm = pd.DataFrame(embed(df_picked_norm.iloc[tstart:tend, -1], w))
    X1_norm= X1_norm.iloc[::-1]

    tstart = tstart + L
    tend = tend + L
    X2_norm = pd.DataFrame(embed(df_picked_norm.iloc[tstart:tend, -1], w))
    X2_norm = X2_norm.iloc[::-1]

    U1_norm, s1_norm, V1_norm = np.linalg.svd(X1_norm.astype(np.float64), full_matrices=False)
    U1_norm = U1_norm[:, 0:m]
    U2_norm, s2_norm, V2_norm = np.linalg.svd(X2_norm.astype(np.float64), full_matrices=False)
    U2_norm = U2_norm[:, 0:m]

    U3_norm, s3_norm, V3_norm = np.linalg.svd(np.dot(U1_norm.T, U2_norm))

    sig1_norm = s3_norm[0]
    abnor_score_norm[t] = 1 - (sig1_norm * sig1_norm)


avg_norm = np.mean(abnor_score_norm)
std_norm = np.std(abnor_score_norm)
th_norm = avg_norm + 0.5*std_norm

change_norm = abnor_score_norm[abnor_score_norm >= th_norm]
change_norm_df = pd.DataFrame(change_norm, columns=['change point_norm_arm'])

print(t)
print("...X1_norm...")
print(X1_norm.head())
print("...X2_norm...")
print(X2_norm.head())
X1_norm.to_csv("CSV_files/X1_norm"+ now +".csv")
X2_norm.to_csv("CSV_files/X2_norm"+ now +".csv")
abnor_score_df_norm = pd.DataFrame(abnor_score_norm, columns=['abnormal_norm_arm'])
data_and_abnor_norm = pd.concat([df_picked_norm, abnor_score_df_norm], axis=1)
#data_and_abnor = pd.concat([df.loc[:, ['instant', 'cnt']], abnor_score_df, df_abnor_check], axis=1)

data_and_abnor_complete = pd.concat([data_and_abnor, data_and_abnor_y, data_and_abnor_z, data_and_abnor_norm], axis=1)
data_and_abnor_complete.to_csv("CSV_files_variation/data_and_abnor_arm"+ now +".csv", index=False)

change_complete = pd.concat([change_x_df, change_y_df, change_z_df, change_norm_df], axis=1)
change_complete.to_csv("CSV_files_variation/change_point_arm"+ now +".csv", index=False)

#t = df[df['sensor_id'] == '810DA3B9'].loc[:, ["ts"]].astype(np.float64)


fig, ax1 = plt.subplots()
ax1.set_xlabel('ts')
#ax1.set_ylabel("Acceleration_x")
#ax1.plot(df_picked["x"].astype(np.float64), color='#377ed8', label="LF/HF")
ax2 = ax1.twinx()
ax2.set_ylabel("abnormality")
ax2.plot(abnor_score_df.values, color='#ff7f00', label="abnormal_acc_x")
ax2.plot(abnor_score_df_y.values, color='g', label="abnormal_acc_y")
ax2.plot(abnor_score_df_z.values, color='#1F77B4', label="abnormal_acc_z")
ax2.plot(abnor_score_df_norm.values, color='#8041D9', label="abnormal_acc_norm")
plt.grid(True)
plt.legend(loc='upper right')
plt.savefig("PNG_files_variation/abnormalscore_arm"+ now +".png")
program_finish = time.time()

elapsed_time = program_finish - program_start
print(f"Execution time:{elapsed_time}sec")