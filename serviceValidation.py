from statistics import mean
import numpy as np
import xlrd
from collections import defaultdict
import pandas as pd


data_service_time = xlrd.open_workbook("Hospital.xlsx")
outcome_normal_service_time = xlrd.open_workbook("outcome_Normal.xlsx")
outcome_extra_service_time = xlrd.open_workbook("outcome_Extra.xlsx")

real_data = data_service_time.sheet_by_name("serving t (length of stay)")
normal_data = outcome_normal_service_time.sheet_by_name("queue_info")
extra_data = outcome_extra_service_time.sheet_by_name("queue_info")



real_service_dict =  {}
normal_service_dict = {}
extra_service_dict = {}

avg_real_service_dict=  {}
avg_normal_service_dict=  {}
avg_extra_service_dict=  {}

for row in range(real_data.nrows):
    if row > 0:
        _data = real_data.row_slice(row)
        if _data[2].value not in real_service_dict:
            real_service_dict[_data[2].value] = [_data[3].value]
        else:
            real_service_dict[_data[2].value].append(_data[3].value)

for row in range(normal_data.nrows):
    if row > 0:
        _data = normal_data.row_slice(row)
        if _data[8].value not in normal_service_dict:
            normal_service_dict[_data[8].value] = [float(_data[13].value)]
        else:
            normal_service_dict[_data[8].value].append(float(_data[13].value))

for row in range(extra_data.nrows):
    if row > 0:
        _data = extra_data.row_slice(row)
        if _data[8].value not in extra_service_dict:
            extra_service_dict[_data[8].value] = [float(_data[13].value)]
        else:
            extra_service_dict[_data[8].value].append(float(_data[13].value))



for _key,_val in real_service_dict.items():
    avg_real_service_dict[_key]=np.nanmean(_val)

for _key,_val in normal_service_dict.items():
    avg_normal_service_dict[_key]=np.nanmean(_val)

for _key,_val in extra_service_dict.items():
    avg_extra_service_dict[_key]=np.nanmean(_val)

#print(avg_real_service_dict)
#print(avg_normal_service_dict)
#print(avg_extra_service_dict)

for k,v in avg_real_service_dict.items():
    print(k)
    print("average from real service time:")
    print("{:.8f}".format(avg_real_service_dict[k]))
    if k in avg_normal_service_dict.keys():
        print("average from normal service time:")
        print("{:.8f}".format(avg_normal_service_dict[k]))
    if k in avg_extra_service_dict.keys():
        print("average from extra service time:")
        print("{:.8f}".format(avg_extra_service_dict[k]))
    print("-----------------------------------------------------")
