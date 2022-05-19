import numpy as np
import pandas as pd
import sys
import time

# resampling to 0.5s

df = pd.read_csv('yoshida_accel3.csv')

df_1 = df[df['sensor_id'] == '810DBA02'] # arm
df_1.reset_index(drop=True, inplace=True)
df_2 = df[df['sensor_id'] == '810DB279'] # neck
df_2.reset_index(drop=True, inplace=True)
df_3 = df[df['sensor_id'] == '810DB2D0'] # waist
df_3.reset_index(drop=True, inplace=True)

size = 2
average_list = list()
average_list_y = list()
average_list_z = list()

for i in range (0, int(len(df_1)/2)):
    df_1_picked = df_1.iloc[size*i:size*i+2,2]
    df_1_picked_y = df_1.iloc[size*i:size*i+2,3]
    df_1_picked_z = df_1.iloc[size*i:size*i+2,4]
    
    list_picked = df_1_picked.tolist()
    list_picked_y = df_1_picked_y.tolist()
    list_picked_z = df_1_picked_z.tolist()
    
    result_sum = sum(list_picked)
    result_sum_y = sum(list_picked_y)
    result_sum_z = sum(list_picked_z)
    
    result = float(result_sum) / size
    result_y = float(result_sum_y) / size
    result_z = float(result_sum_z) / size
    
    average_list.append(result)
    average_list_y.append(result_y)
    average_list_z.append(result_z)

average_df_x = pd.DataFrame(average_list, columns=['x_mean'])
average_df_y = pd.DataFrame(average_list_y, columns=['y_mean'])
average_df_z = pd.DataFrame(average_list_z, columns=['z_mean'])

average_df_arm = pd.concat([average_df_x, average_df_y, average_df_z], axis=1)

size = 5
average_list = list()
average_list_y = list()
average_list_z = list()

for i in range (0, int(len(df_2)/5)):
    df_2_picked = df_2.iloc[size*i:size*i+5,2]
    df_2_picked_y = df_2.iloc[size*i:size*i+5,3]
    df_2_picked_z = df_2.iloc[size*i:size*i+5,4]
    
    list_picked = df_2_picked.tolist()
    list_picked_y = df_2_picked_y.tolist()
    list_picked_z = df_2_picked_z.tolist()
    
    result_sum = sum(list_picked)
    result_sum_y = sum(list_picked_y)
    result_sum_z = sum(list_picked_z)
    
    result = float(result_sum) / size
    result_y = float(result_sum_y) / size
    result_z = float(result_sum_z) / size
    
    average_list.append(result)
    average_list_y.append(result_y)
    average_list_z.append(result_z)

average_df_x = pd.DataFrame(average_list, columns=['x_mean'])
average_df_y = pd.DataFrame(average_list_y, columns=['y_mean'])
average_df_z = pd.DataFrame(average_list_z, columns=['z_mean'])

average_df_neck = pd.concat([average_df_x, average_df_y, average_df_z], axis=1)

size = 5 
average_list = list()
average_list_y = list()
average_list_z = list()

for i in range (0, int(len(df_3)/5)):
    df_3_picked = df_3.iloc[size*i:size*i+5,2]
    df_3_picked_y = df_3.iloc[size*i:size*i+5,3]
    df_3_picked_z = df_3.iloc[size*i:size*i+5,4]
    
    list_picked = df_3_picked.tolist()
    list_picked_y = df_3_picked_y.tolist()
    list_picked_z = df_3_picked_z.tolist()
    
    result_sum = sum(list_picked)
    result_sum_y = sum(list_picked_y)
    result_sum_z = sum(list_picked_z)
    
    result = float(result_sum) / size
    result_y = float(result_sum_y) / size
    result_z = float(result_sum_z) / size
    
    average_list.append(result)
    average_list_y.append(result_y)
    average_list_z.append(result_z)

average_df_x = pd.DataFrame(average_list, columns=['x_mean'])
average_df_y = pd.DataFrame(average_list_y, columns=['y_mean'])
average_df_z = pd.DataFrame(average_list_z, columns=['z_mean'])

average_df_waist = pd.concat([average_df_x, average_df_y, average_df_z], axis=1)

average_df_arm.to_csv("yoshida_accel3_arm_mean_5.csv", index=False)
average_df_neck.to_csv("yoshida_accel3_neck_mean_5.csv", index=False)
average_df_waist.to_csv("yoshida_accel3_waist_mean_5.csv", index=False)