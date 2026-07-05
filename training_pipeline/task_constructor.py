import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

from typing import Callable, List, Tuple
from dataclasses import dataclass, field

from training_pipeline.metric_calculators import (
    MetricCalculator,
    ChurnMetricCalculator,
    PropensityMetricCalculator,
)
from training_pipeline.tasks import (
    ValidTasks,
    ChurnTasks,
    PropensityTasks,
    TaskNotSupportedError,
)
from training_pipeline.target_calculators import (
    TargetCalculator,
    ChurnTargetCalculator,
    PropensityTargetCalculator,
)

from data_utils.data_dir import DataDir
from training_pipeline.metrics_containers import (
    MetricContainer,
)


@dataclass(frozen=True)
class TaskSettings:
    """
    Container class which stores all task specific data structures.
    """

    target_calculator: TargetCalculator
    metric_calculator: MetricCalculator
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    metrics_tracker: List[MetricContainer] = field(default_factory=list)


class TaskConstructor:
    """
    Class for constructing all task specific data structures.
    """

    def __init__(self, data_dir: DataDir):
        """
        Args:
            data_dir (DataDir): container for simplified access to subdirectories of data_dir.
        """
        self.data_dir = data_dir

    def construct_task(self, task: ValidTasks) -> TaskSettings:
        """
        Method for constructing task specific data structures.

        Args:
            task (ValidTasks): task for which data structures are constructed.
        Returns:
            TaskSettings: container with data structures for given task
        """
        if isinstance(task, ChurnTasks):
            return self._construct_churn_task(task=task)
        elif isinstance(task, PropensityTasks):
            return self._construct_propensity_task(task=task)
        else:
            raise TaskNotSupportedError("An unsupported task was provided.")

    def _construct_churn_task(self, task: ChurnTasks) -> TaskSettings:
        target_calculator = ChurnTargetCalculator()
        metric_calculator = ChurnMetricCalculator()

        n_pos = 6010.0   # target == 1 (Mehrheit: kein Re-Kauf)
        n_neg = 161.0    # target == 0 (Minderheit: Re-Kauf)
        total = n_pos + n_neg
        w1 = total / (2.0 * n_pos)   # ≈ 0.513
        w0 = total / (2.0 * n_neg)   # ≈ 19.16

        def weighted_bce(pred, target):
            weight = torch.where(
                target == 1,
                torch.full_like(target, w1),
                torch.full_like(target, w0),
            )
            return F.binary_cross_entropy_with_logits(pred, target, weight=weight)

        return TaskSettings(
            target_calculator=target_calculator,
            metric_calculator=metric_calculator,
            loss_fn=weighted_bce,
        )
    
    def _compute_propensity_pos_weight(
        self, task: PropensityTasks, propensity_targets: np.ndarray
    ) -> torch.Tensor:
        """
        Ein pos_weight PRO Klasse (nicht ein einzelner Skalar wie bei Churn),
        weil jede der 20 Kategorien/SKUs unterschiedlich oft positiv ist.
        """
        from training_pipeline.target_data import TargetData

        target_data = TargetData.read_from_dir(target_dir=self.data_dir.target_dir)
        all_targets = pd.concat([target_data.train_df, target_data.validation_df])

        column = task.value  # z.B. "propensity_category"
        n_total = len(all_targets)

        exploded = (
            all_targets[["client_id", column]]
            .explode(column)
            .dropna(subset=[column])
        )
        counts_series = exploded[column].value_counts()

        counts = np.array(
            [counts_series.get(cls_id, 0) for cls_id in propensity_targets],
            dtype=np.float64,
        )
        counts = np.clip(counts, 1, None)  # Division durch 0 verhindern

        weights = (n_total - counts) / counts
        weights = np.clip(weights, 1.0, 20.0)  # Obergrenze gegen Instabilität

        return torch.tensor(weights, dtype=torch.float32)
    def _construct_propensity_task(self, task: PropensityTasks) -> TaskSettings:
        propensity_targets, popularity_data = self._load_propensity_targets(task)

        target_calculator = PropensityTargetCalculator(
            task=task, propensity_targets=propensity_targets
        )

        pos_weight = self._compute_propensity_pos_weight(task, propensity_targets)

        def weighted_bce(pred, target):
            return F.binary_cross_entropy_with_logits(
                pred, target, pos_weight=pos_weight.to(pred.device)
            )

        metric_calculator = PropensityMetricCalculator(
            output_dim=target_calculator.target_dim,
            popularity_data=popularity_data,
        )

        return TaskSettings(
            target_calculator=target_calculator,
            metric_calculator=metric_calculator,
            loss_fn=weighted_bce,
        )

    def _load_propensity_targets(
        self,
        task: PropensityTasks,
    ) -> Tuple[np.ndarray, np.ndarray]:

        propensity_targets = np.load(
            self.data_dir.target_dir / f"{task.value}.npy",
            allow_pickle=True,
        )
        popularity_data = np.load(
            self.data_dir.target_dir / f"popularity_{task.value}.npy"
        )

        return propensity_targets, popularity_data


def transform_client_ids_and_embeddings(
    task: ValidTasks,
    client_ids: np.ndarray,
    embeddings: np.ndarray,
    data_dir: DataDir,
) -> Tuple[np.ndarray, np.ndarray]:
    if task == ChurnTasks.CHURN:
        from training_pipeline.target_data import TargetData
        import pandas as pd

        target_data = TargetData.read_from_dir(target_dir=data_dir.target_dir)
        all_targets = pd.concat([target_data.train_df, target_data.validation_df])
        churn_defined_ids = all_targets.loc[
            all_targets["churn"].notna(), "client_id"
        ].to_numpy()

        mask = np.isin(client_ids, churn_defined_ids)
        return client_ids[mask], embeddings[mask]
    return client_ids, embeddings
