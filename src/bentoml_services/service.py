import numpy as np
import bentoml
from bentoml.io import NumpyNdarray
from sidetrek import load_custom_objects


model_runner = bentoml.sklearn.get("sepsis_model:latest").to_runner()
svc = bentoml.Service("sepsis_model", runners=[model_runner]) # it's important that this is assinged to a variable named "svc"
latest_model = bentoml.sklearn.get("sepsis_model:latest").tag
scaler = load_custom_objects(f"{latest_model}", "scaler_object")


@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def sepsis_model(input_series: np.ndarray) -> np.ndarray:
    input_series = scaler.transform(input_series)
    result = model_runner.predict.run(input_series)
    return result