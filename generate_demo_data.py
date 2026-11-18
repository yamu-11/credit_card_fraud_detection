
import numpy as np
import pandas as pd
import os

# Generate synthetic transaction data
np.random.seed(42)

n = 5000

# Generate transaction classes
fraud = np.random.choice([0, 1], size=n, p=[0.98, 0.02])

data = {
    "Time": np.random.uniform(0, 172800, n),
    "Amount": np.random.lognormal(3, 1, n)
}

# Generate anonymized features
for i in range(1, 29):
    shift = fraud * np.random.choice([-1, 1]) * 0.5
    data[f"V{i}"] = np.random.normal(0, 1, n) + shift

data["Class"] = fraud

df = pd.DataFrame(data)

os.makedirs("data", exist_ok=True)

df.to_csv("data/demo_creditcard.csv", index=False)

print("Demo dataset generated successfully!")
print(df.head())
print("\nDataset shape:", df.shape)
print("\nFraud transactions:", df["Class"].sum())
