#STEP 1 Checking how many columns are there in the csv file

import pandas as pd
# Load Netflix dataset
df = pd.read_csv("d1.csv")
# Show first 5 rows
print(df.head())
# Show column names
print(df.columns)
print("Total rows:", len(df))



