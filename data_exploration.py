import pandas as pd
from collections import Counter

# simple script to print distribution of classes
data_set = pd.read_csv("../data/train.csv")
counter = Counter(data_set['label'])
print(counter)
