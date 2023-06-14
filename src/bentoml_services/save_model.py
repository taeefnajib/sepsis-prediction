"""
THIS IS JUST A TEMPLATE - CHANGE IT TO FIT YOUR NEEDS
"""

import joblib
import bentoml


with open("/userRepoData/__sidetrek__/taeefnajib/sepsis-prediction/bentoml/models/7f320545a41e2f303278a871c014baa1.joblib", "rb") as f:
    model = joblib.load(f)
    saved_model = bentoml.sklearn.save_model(
        "sepsis_model",
        model,
    )
    print(saved_model) # This is required!