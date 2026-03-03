import pandas as pd

# Load your dataset file
df = pd.read_csv("d1.csv")  # <-- replace with your actual filename

# Get unique diseases
diseases = df["Disease"].unique()

print("Total Diseases:", len(diseases))
print("\nDisease List:\n")

for d in diseases:
    print(d)