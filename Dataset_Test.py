import pandas as pd

df = pd.read_csv("data/Combined.csv")

print(df["status"].value_counts())

