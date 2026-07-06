import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

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
    ActiveTargetCalculator,
    ChurnTargetCalculator,
    PropensityTargetCalculator,
)
from training_pipeline.target_data import TargetData

from data_utils.data_dir import DataDir
from training_pipeline.metrics_containers import MetricContainer


@dataclass(frozen=True)
class TaskSettings:
    target_calculator: TargetCalculator
    metric_calculator: MetricCalculator
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    metrics_tracker: List[MetricContainer] = field(default_factory=list)


class TaskConstructor:
    def __init__(self, data_dir: DataDir):
        self.data_dir = data_dir

    def construct_task(self, task: ValidTasks) -> TaskSettings:
        if isinstance(task, ChurnTasks):
            return self._construct_churn_task(task=task)
        elif isinstance(task, PropensityTasks):
            return self._construct_propensity_task(task=task)
        else:
            raise TaskNotSupportedError("An unsupported task was provided.")

    def _load_all_targets(self) -> pd.DataFrame:
        target_data = TargetData.read_from_dir(target_dir=self.data_dir.target_dir)
        return pd.concat(
            [target_data.train_df, target_data.validation_df], ignore_index=True
        )

    def _construct_churn_task(self, task: ChurnTasks) -> TaskSettings:
        all_targets = self._load_all_targets()

        if task == ChurnTasks.ACTIVE:
            # Active ist deutlich ausgeglichener (~14% positiv vs. <1% bei
            # Kauf-basierten Tasks) - keine spezielle Klassengewichtung noetig.
            target_calculator = ActiveTargetCalculator(target_df=all_targets)
            metric_calculator = ChurnMetricCalculator()
            return TaskSettings(
                target_calculator=target_calculator,
                metric_calculator=metric_calculator,
                loss_fn=F.binary_cross_entropy_with_logits,
            )

        target_calculator = ChurnTargetCalculator(target_df=all_targets)
        metric_calculator = ChurnMetricCalculator()

        churn_defined = all_targets["churn"].dropna()
        n_pos = float((churn_defined == 1).sum())  # Mehrheit: kein Re-Kauf
        n_neg = float((churn_defined == 0).sum())  # Minderheit: Re-Kauf
        total = n_pos + n_neg

        # Symmetrische Klassen-Gewichtung (NICHT pos_weight, das nur target==1
        # skaliert -- waere hier falsch, weil Label 1 bei Churn die MEHRHEIT ist).
        w1 = total / (2.0 * max(n_pos, 1.0))
        w0 = total / (2.0 * max(n_neg, 1.0))

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

    def _construct_propensity_task(self, task: PropensityTasks) -> TaskSettings:
        propensity_targets, popularity_data = self._load_propensity_targets(task)
        all_targets = self._load_all_targets()

        target_calculator = PropensityTargetCalculator(
            task=task, propensity_targets=propensity_targets, target_df=all_targets
        )

        pos_weight = self._compute_propensity_pos_weight(task, propensity_targets, all_targets)

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

    def _compute_propensity_pos_weight(
        self,
        task: PropensityTasks,
        propensity_targets: np.ndarray,
        all_targets: pd.DataFrame,
    ) -> torch.Tensor:
        column = task.value
        n_total = len(all_targets)

        exploded = all_targets[["client_id", column]].explode(column).dropna(subset=[column])
        counts_series = exploded[column].value_counts()

        counts = np.array(
            [counts_series.get(cls_id, 0) for cls_id in propensity_targets],
            dtype=np.float64,
        )
        counts = np.clip(counts, 1, None)

        weights = (n_total - counts) / counts
        weights = np.clip(weights, 1.0, 20.0)  # Cap gegen Instabilitaet

        return torch.tensor(weights, dtype=torch.float32)

    def _load_propensity_targets(self, task: PropensityTasks) -> Tuple[np.ndarray, np.ndarray]:
        propensity_targets = np.load(
            self.data_dir.target_dir / f"{task.value}.npy", allow_pickle=True
        )
        popularity_data = np.load(self.data_dir.target_dir / f"popularity_{task.value}.npy")
        return propensity_targets, popularity_data


def transform_client_ids_and_embeddings(
    task: ValidTasks,
    client_ids: np.ndarray,
    embeddings: np.ndarray,
    data_dir: DataDir,
) -> Tuple[np.ndarray, np.ndarray]:
    if task == ChurnTasks.CHURN:
        target_data = TargetData.read_from_dir(target_dir=data_dir.target_dir)
        all_targets = pd.concat([target_data.train_df, target_data.validation_df])
        churn_defined_ids = all_targets.loc[
            all_targets["churn"].notna(), "client_id"
        ].to_numpy()
        mask = np.isin(client_ids, churn_defined_ids)
        return client_ids[mask], embeddings[mask]
    return client_ids, embeddings