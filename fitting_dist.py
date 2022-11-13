import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats
import xlrd
from fitter import Fitter, get_common_distributions, get_distributions

distribution_per_wards = {}
# dataset2 = pd.read_csv("weight_height.csv")


orig_dataset = pd.read_excel("akademiska.xlsx", "serving t (length of stay)")
orig_dataset = orig_dataset.dropna(subset=['los_ward'])

file = xlrd.open_workbook("akademiska.xlsx")  # access to the file
wards = file.sheet_by_name("Nodes")  # access to nodes attributes

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
    except ValueError:
        print("Ignoring Value error: len(serving_time)=" + str(len(serving_time)))



for ward in wards_list:
    try:
        dist_name = list(distribution_per_wards[ward])[0]
        args = distribution_per_wards[ward][dist_name]

        dist = getattr(scipy.stats, dist_name)

        vals = dist.rvs(**args, size=1000)
        vals_df = pd.DataFrame(vals, columns=[dist_name])
        vals_df.to_excel("fitted_distributions_" + str(ward) + ".xlsx", sheet_name='wards_dist')
    except KeyError:
        print("2 - Ignoring Value error: len(serving_time)=" + str(len(serving_time)))

#sns.set_style('white')
#sns.set_context("paper", font_scale=2)

#vals_df2 = vals_df[vals_df['los_ward'] < float(dataset.max())]
#sns.displot(data=vals_df2, x="los_ward", kind="hist", bins=100, aspect=1.5)


