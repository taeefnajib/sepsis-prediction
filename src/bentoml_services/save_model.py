"""
THIS IS JUST A TEMPLATE - CHANGE IT TO FIT YOUR NEEDS
"""

import joblib
import bentoml


with open("/userRepoData/__sidetrek__/taeefnajib/sepsis-prediction/bentoml/models/a8258e43f26a21093b653a41089efb88.joblib", "rb") as f:
    model = joblib.load(f)
    saved_model = bentoml.sklearn.save_model(
        "sepsis_2",
        model,
    )
    print(saved_model) # This is required!