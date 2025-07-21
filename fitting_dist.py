import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats
import xlrd
from fitter import Fitter, get_common_distributions, get_distributions

distribution_per_wards = {}


orig_dataset = pd.read_excel("akademiska.xlsx", "serving t (length of stay)")
orig_dataset = orig_dataset.dropna(subset=['los_ward'])

file = xlrd.open_workbook("akademiska.xlsx") 
wards = file.sheet_by_name("Nodes")  

wards_list = []

for row in range(wards.nrows):
    if row > 0:
        _data = wards.row_slice(row)
        wards_list.append(_data[0].value)


print(wards_list)

for ward in wards_list:
    dataset = orig_dataset.loc[orig_dataset['OA_unit_SV'] == ward]
    dataset = dataset[['los_ward']]
    dataset.info()
    sns.set_style('white')
    sns.set_context("paper", font_scale=2)
    sns.displot(data=dataset, x="los_ward", kind="hist", bins=100, aspect=1.5)
    serving_time = dataset["los_ward"].values

    try:
        f = Fitter(serving_time,distributions=get_distributions())
        f.fit()
        f.summary()
        distribution_per_wards[ward] = f.get_best(method='sumsquare_error')
        pd.DataFrame(distribution_per_wards[ward]).to_excel("fitted_distributions_" + str(ward) + ".xlsx", sheet_name='wards_dist')
    except ValueError:
        print("Ignoring Value error: len(serving_time)=" + str(len(serving_time)))


