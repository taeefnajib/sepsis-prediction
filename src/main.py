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
from sklearn.preprocessing import MinMaxScaler
from dataclasses_json import dataclass_json
from dataclasses import dataclass
from typing import Tuple
from sidetrek.types.dataset import SidetrekDataset
from sidetrek.dataset import load_dataset
from sidetrek import save_custom_objects
 

@dataclass_json
@dataclass
class Hyperparameters(object):
    test_size: float = 0.2
    random_state: int = 6
    lr_max_iter: int = 500

# Instantiating Hyperparameters class
hp = Hyperparameters

def get_data(ds: SidetrekDataset) -> pd.DataFrame:
    data = load_dataset(ds=ds, data_type="csv")
    data_dict = {}
    cols = list(data)[0]
    cols_object = save_custom_objects(key="cols_object", value=cols)
    for k,v in enumerate(data):
        if k>0:
            data_dict[k]=v 
    
    df = pd.DataFrame.from_dict(data_dict, columns=cols, orient="index")
    type_dict = {
        "PRG" : int,
        "PL" : int,
        "PR" : int,
        "SK" : int,
        "TS" : int,
        "M11" : float,
        "BD2" : float,
        "Age" : int,
        "Insurance" : int,
    }
    df = df.astype(type_dict)
    return df

def preprocess_ds(df: pd.DataFrame)->pd.DataFrame:
    df.drop(["\ufeffID"], axis=1, inplace=True)
    df = pd.get_dummies(df, columns=["Sepssis"],drop_first=True)
    return df

def split_ds(hp: Hyperparameters, df: pd.DataFrame)-> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    X = df.drop(["Sepssis_Positive"], axis=1)
    scaler = MinMaxScaler()
    scaler.fit(X)
    scaler_object = save_custom_objects(key="scaler_object", value=scaler)
    X = scaler.transform(X)
    y = df["Sepssis_Positive"]
    return train_test_split(X, y, test_size = hp.test_size, random_state = hp.random_state)



def train_model(hp: Hyperparameters, X_train: np.ndarray, y_train: pd.Series) -> VotingClassifier:
    lr = LogisticRegression(max_iter=hp.lr_max_iter, random_state=hp.random_state)
    knn = KNeighborsClassifier()
    svc = SVC(random_state=hp.random_state, probability=True)
    dt = DecisionTreeClassifier(random_state=hp.random_state)
    model = VotingClassifier(estimators=[('lr', lr), ('knn', knn), ('svc', svc), ('dt', dt)], voting='soft')
    return model.fit(X_train,y_train)


# def main(hp: Hyperparameters):
#     df = get_data(source="data/train.csv")
#     df = preprocess_ds(df=df)
#     X_train, X_test, y_train, y_test = split_ds(hp=hp, df=df)
#     return train_model(hp=hp, X_train=X_train, y_train=y_train)


# def get_accuracy() -> None:
#     model = train_model(hp=hp)
#     X_train, X_test, y_train, y_test = get_data(hp=hp)
#     y_pred = model.predict(X_test)
#     acc = accuracy_score(y_test, y_pred)
#     print(f"Accuracy: {acc}")

# if __name__=="__main__":
#     print(main(hp=hp))