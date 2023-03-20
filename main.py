# Importing all dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from dataclasses_json import dataclass_json
from dataclasses import dataclass
from typing import List, Union


@dataclass_json
@dataclass
class Hyperparameters(object):
    data_filepath: str = "data/train.csv"
    test_size: float = 0.2
    random_state: int = 6
    lr_max_iter: int = 500

# Instantiating Hyperparameters class
hp = Hyperparameters

def get_data(hp: Hyperparameters) -> List[Union[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
    df = pd.read_csv(hp.data_filepath)
    df.drop(["ID"], axis=1, inplace=True)
    df = pd.get_dummies(df, columns=["Sepssis"],drop_first=True)
    X = df.drop(["Sepssis_Positive"], axis=1)
    y = df["Sepssis_Positive"]
    return train_test_split(X, y, test_size = hp.test_size, random_state = hp.random_state)

def train_model(hp: Hyperparameters) -> VotingClassifier:
    X_train, X_test, y_train, y_test = get_data(hp=hp)
    lr = LogisticRegression(max_iter=hp.lr_max_iter, random_state=hp.random_state)
    knn = KNeighborsClassifier()
    svc = SVC(random_state=hp.random_state, probability=True)
    dt = DecisionTreeClassifier(random_state=hp.random_state)
    model = VotingClassifier(estimators=[('lr', lr), ('knn', knn), ('svc', svc), ('dt', dt)], voting='soft')
    return model.fit(X_train,y_train)

# def get_accuracy() -> None:
#     model = train_model(hp=hp)
#     X_train, X_test, y_train, y_test = get_data(hp=hp)
#     y_pred = model.predict(X_test)
#     acc = accuracy_score(y_test, y_pred)
#     print(f"Accuracy: {acc}")

if __name__=="__main__":
    train_model(hp=hp)
    



