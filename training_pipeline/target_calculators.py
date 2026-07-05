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
    build_retailrocket_eval_dataset.py, instead of assuming presence/
    absence in an event-level table.
    """

    @property
    def target_dim(self) -> int:
        return 1

    def compute_target(self, client_id: int, target_df: pd.DataFrame) -> np.ndarray:
        target = np.zeros(self.target_dim, dtype=np.float32)
        rows = target_df.loc[target_df["client_id"] == client_id]
        if rows.empty:
            return target
        value = rows["churn"].iloc[0]
        target[0] = 0.0 if pd.isna(value) else float(value)
        return target


class PropensityTargetCalculator(TargetCalculator):
    """
    Reads list-valued columns propensity_category / propensity_sku /
    propensity_new_sku directly (one row per client, not one row per
    purchase event).
    """

    def __init__(self, task: PropensityTasks, propensity_targets: np.ndarray):
        self._column = task.value  # matches our schema exactly, e.g. "propensity_category"
        self._propensity_targets = propensity_targets

    @property
    def target_dim(self) -> int:
        return len(self._propensity_targets)

    def compute_target(self, client_id: int, target_df: pd.DataFrame) -> np.ndarray:
        target = np.zeros(self.target_dim, dtype=np.float32)
        rows = target_df.loc[target_df["client_id"] == client_id]
        if rows.empty:
            return target
        positive_ids = rows[self._column].iloc[0]
        if positive_ids is None or len(positive_ids) == 0:
            return target
        positive_ids = np.asarray(positive_ids)
        target[np.isin(self._propensity_targets, positive_ids, assume_unique=True)] = 1.0
        return target