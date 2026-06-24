import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import os

os.makedirs("models", exist_ok=True)

def train_model(file, name):

    df = pd.read_csv(f"datasets/{file}.csv")

    X = df.drop("target", axis=1)
    y = df["target"]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    rf = RandomForestClassifier(n_estimators=150)
    gb = GradientBoostingClassifier()
    svm = SVC(probability=True)
    mlp = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500)

    stack = StackingClassifier(
        estimators=[
            ("rf", rf),
            ("gb", gb),
            ("svm", svm),
            ("mlp", mlp)
        ],
        final_estimator=LogisticRegression(max_iter=2000)
    )

    stack.fit(X_train, y_train)

    joblib.dump(stack, f"models/{name}_model.pkl")
    joblib.dump(scaler, f"models/{name}_scaler.pkl")

    print(name, "trained")

train_model("diabetes", "diabetes")
train_model("heart", "heart")
train_model("parkinsons", "parkinsons")
train_model("liver", "liver")

print("All models trained!")
