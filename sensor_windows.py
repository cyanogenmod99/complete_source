import sys
import serial
import datetime
import time
import json
import numpy as np
import pandas as pd

ser = serial.Serial(port='COM6', baudrate=115200, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, bytesize=serial.EIGHTBITS, timeout=1)
now = str(int(time.time()))

# make file
acc_log_path = './test_csv' + now + '.csv'  #change name

f = open(acc_log_path, 'x', newline='')

f.write('ts,datetime,sensor_id,x,y,z\n')

s_key = ''
s_val = ''

ts = 0
sensor_id=''
x = y = z = 0.0
now6 = datetime.datetime.now()
td = datetime.timedelta(milliseconds=10)

start_time = time.time()

now = datetime.datetime.now()
td = datetime.timedelta(milliseconds=10)


temp = 0
temp1 = {}


try:
    while 1:
        data = ser.readline()

        # 先頭の「:」を取り除く
        data = data[2:]
        spilitdatum = data.decode('utf-8').split(":")
        
        dict = {}

        # key,value型にしてディクショナリに保存
        for spilitdata in spilitdatum:

            s_key = spilitdata.split("=")[0]
            s_val = spilitdata.split("=")[1]

            dict[s_key] = s_val
            
        if(len(dict) < 2):
            temp = float(dict['ts'])
        
        else:
            temp1 = dict
            #temp += 0.01
        now = datetime.datetime.now()
        #print(temp, dict) 

        
        #print(type(temp), type(temp1['ed']), type(temp1['x']), type(temp1['ed']), type(temp1['x']), type(temp1['y']), type(temp1['z']))
        new_data = '{:.2f},{},{},{},{},{}\n'.format(temp, now, temp1['ed'], int(temp1['x']), int(temp1['y']), int(temp1['z']))
        f.write(new_data)

        #print(new_data)

except KeyboardInterrupt:
    print('\nInterrupted')

f.close()

# time
tact_time = int(time.time() - start_time)
if tact_time >= 60:
    minutes = int(tact_time // 60)
    seconds = int(tact_time % 60)
    tact_time = '{}min {}sec'.format(minutes, seconds)
else:
    tact_time = '{}sec'.format(tact_time)
print('tact_time:{}'.format(tact_time))

ser.close()
sys.exit(0)