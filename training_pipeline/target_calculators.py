import numpy as np
import pandas as pd

from abc import ABC, abstractmethod, abstractproperty
from training_pipeline.tasks import PropensityTasks


class TargetCalculator(ABC):
    @abstractproperty
    def target_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def compute_target(self, client_id: int, target_df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class ChurnTargetCalculator(TargetCalculator):
    """
    Reads the explicit `churn` column (nullable Int8) produced by
    build_retailrocket_eval_dataset.py. Uses a dict lookup (built once
    at construction) instead of scanning the DataFrame on every sample.
    """

    def __init__(self, target_df: pd.DataFrame):
        self._churn_by_client = dict(zip(target_df["client_id"], target_df["churn"]))

    @property
    def target_dim(self) -> int:
        return 1

    def compute_target(self, client_id: int, target_df: pd.DataFrame) -> np.ndarray:
        target = np.zeros(self.target_dim, dtype=np.float32)
        value = self._churn_by_client.get(client_id)
        if value is not None and not pd.isna(value):
            target[0] = float(value)
        return target


class PropensityTargetCalculator(TargetCalculator):
    """
    Reads list-valued columns propensity_category / propensity_sku /
    propensity_new_sku directly (one row per client, not one row per
    purchase event). Dict lookup built once at construction.
    """

    def __init__(
        self,
        task: PropensityTasks,
        propensity_targets: np.ndarray,
        target_df: pd.DataFrame,
    ):
        self._column = task.value  # entspricht exakt unserem Schema
        self._propensity_targets = propensity_targets
        self._labels_by_client = dict(zip(target_df["client_id"], target_df[self._column]))

    @property
    def target_dim(self) -> int:
        return len(self._propensity_targets)

    def compute_target(self, client_id: int, target_df: pd.DataFrame) -> np.ndarray:
        target = np.zeros(self.target_dim, dtype=np.float32)
        positive_ids = self._labels_by_client.get(client_id)
        if positive_ids is not None and len(positive_ids) > 0:
            positive_ids = np.asarray(positive_ids)
            target[np.isin(self._propensity_targets, positive_ids, assume_unique=True)] = 1.0
        return target