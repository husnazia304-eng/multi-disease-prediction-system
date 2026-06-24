import numpy as np
import pandas as pd
import os

os.makedirs("datasets", exist_ok=True)

def save(df, name):
    df.to_csv(f"datasets/{name}.csv", index=False)

# DIABETES
def diabetes():
    n = 1000
    df = pd.DataFrame({
        "glucose": np.random.randint(70, 200, n),
        "bmi": np.random.uniform(18, 40, n),
        "age": np.random.randint(20, 80, n),
        "bp": np.random.randint(60, 140, n),
        "insulin": np.random.randint(15, 300, n),
    })
    df["target"] = ((df.glucose > 130) & (df.bmi > 30)).astype(int)
    save(df, "diabetes")

# HEART
def heart():
    n = 1000
    df = pd.DataFrame({
        "age": np.random.randint(25, 80, n),
        "cholesterol": np.random.randint(150, 300, n),
        "bp": np.random.randint(80, 180, n),
        "max_hr": np.random.randint(90, 200, n),
        "st": np.random.uniform(0, 5, n),
    })
    df["target"] = ((df.cholesterol > 240) & (df.bp > 140)).astype(int)
    save(df, "heart")

# PARKINSONS
def parkinsons():
    n = 1000
    df = pd.DataFrame({
        "tremor": np.random.uniform(0, 1, n),
        "voice": np.random.uniform(0, 1, n),
        "speech": np.random.uniform(50, 200, n),
        "hand": np.random.uniform(0, 1, n),
        "balance": np.random.uniform(0, 1, n),
    })
    df["target"] = ((df.tremor > 0.6) & (df.balance < 0.4)).astype(int)
    save(df, "parkinsons")

# LIVER
def liver():
    n = 1000
    df = pd.DataFrame({
        "bilirubin": np.random.uniform(0.2, 5.0, n),
        "alk": np.random.randint(100, 500, n),
        "alt": np.random.randint(10, 200, n),
        "ast": np.random.randint(10, 200, n),
        "albumin": np.random.uniform(2, 5, n),
    })
    df["target"] = ((df.bilirubin > 2) & (df.alt > 80)).astype(int)
    save(df, "liver")

diabetes()
heart()
parkinsons()
liver()

print("Datasets created!")
