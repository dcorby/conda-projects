# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

NOTEBOOK = False
input_dir = "../input/digit-recognizer" if NOTEBOOK else "input"


data_train = pd.read_csv(f"{input_dir}/train.csv")
data_test = pd.read_csv(f"{input_dir}/test.csv")

x = data_train.drop(columns="label").to_numpy().reshape(-1, 28, 28, 1)
print("x.shape:")
print(x.shape)
y = data_train["label"]
print("y.shape:")
print(y.shape)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.10, shuffle=True)
print("x_train.shape:")
print(x_train.shape)
print("x_val.shape:")
print(x_val.shape)
print("y_train.shape:")
print(y_train.shape)
print("y_val.shape:")
print(y_val.shape)
