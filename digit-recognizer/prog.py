# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
    

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Referencing this notebook: https://www.kaggle.com/code/dyimahansah/digit-recognizer-competition

NOTEBOOK = False
#import sys
#sys.exit()

input_dir = "../input/digit-recognizer" if NOTEBOOK else "input"

data_train = pd.read_csv(f"{input_dir}/train.csv")
data_test = pd.read_csv(f"{input_dir}/test.csv")

data_test.describe()
data_train.head()
data_train["label"].unique()
print("training data shape:")
print(data_train.shape)
print("testing data shape:")
print(data_test.shape)

import matplotlib.pyplot as plt
plt.hist(data_train["label"], bins=10, rwidth=0.6, align="left")
plt.xticks(range(10))
plt.title("Label Frequency")
plt.ylabel("Frequency")
plt.xlabel("Label")
plt.show()

plt.figure(figsize=(6, 6))  # width and height in inches (not related to subplots)
for num in range(0, 25):
    plt.subplot(5, 5, num + 1)
    np_grid = data_train.iloc[num, 1:].to_numpy().reshape(28, 28)
    plt.imshow(np_grid, interpolation="none", cmap=plt.cm.binary)
    plt.axis('off')
plt.tight_layout()
plt.show()

#raise
#print(type(data_train))

#x = data_train.drop(columns="label").to_numpy().reshape(-1, 28, 28, 1)
#y = data_train["label"]

#print(x[0])

#X_train, x_val, y_train, y_val = train_test_split(X,y, test_size=0.10, shuffle=True)



