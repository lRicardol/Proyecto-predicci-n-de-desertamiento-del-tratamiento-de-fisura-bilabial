"""Contrato atómico del modelo de deserción (ISP: no incluye analítica conversacional)."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class IPredictorModel(ABC):
    """Puerto Strategy: cualquier modelo sustituible debe devolver P(deserción)."""

    @abstractmethod
    def predict_dropout_probability(self, features: pd.DataFrame) -> np.ndarray:
        """Retorna un vector de probabilidades en el intervalo [0.0, 1.0]."""
