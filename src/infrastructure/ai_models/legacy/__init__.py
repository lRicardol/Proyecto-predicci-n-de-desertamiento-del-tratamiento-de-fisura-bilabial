"""Modelos históricos NO usados por la API.

``xgboost_predictor.py`` y ``model.joblib`` corresponden al modelo anterior,
entrenado sobre ``dataset_final_limpio.csv`` con ``citas_perdidas_noshow``
(acumulado del tratamiento: fuga de información para una predicción al ingreso).
Se conservan solo como referencia del antes/después documentado en
``reports/auditoria.md``. No los importe desde ``composition.py`` ni desde la API.
"""
