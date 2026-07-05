import numpy as np
import pytorch_lightning as pl
import logging

from torch.utils.data import DataLoader

from training_pipeline.dataset import BehavioralDataset
from training_pipeline.target_data import TargetData
from training_pipeline.target_calculators import TargetCalculator

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class BehavioralDataModule(pl.LightningDataModule):
    def __init__(
        self,
        embeddings: np.ndarray,
        client_ids: np.ndarray,
        target_data: TargetData,
        target_calculator: TargetCalculator,
        batch_size: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.client_ids = client_ids
        self.embeddings = embeddings
        self.target_data = target_data
        self.target_calculator = target_calculator

    def setup(self, stage) -> None:
        if stage == "fit":
            logger.info("Constructing datasets")

            train_ids = set(self.target_data.train_df["client_id"].tolist())
            valid_ids = set(self.target_data.validation_df["client_id"].tolist())

            train_mask = np.isin(self.client_ids, list(train_ids))
            valid_mask = np.isin(self.client_ids, list(valid_ids))

            logger.info(
                "Train clients: %d, Validation clients: %d (of %d total)",
                train_mask.sum(), valid_mask.sum(), len(self.client_ids),
            )

            self.train_data = BehavioralDataset(
                embeddings=self.embeddings[train_mask],
                client_ids=self.client_ids[train_mask],
                target_df=self.target_data.train_df,
                target_calculator=self.target_calculator,
            )

            self.validation_data = BehavioralDataset(
                embeddings=self.embeddings[valid_mask],
                client_ids=self.client_ids[valid_mask],
                target_df=self.target_data.validation_df,
                target_calculator=self.target_calculator,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.validation_data, batch_size=self.batch_size, num_workers=self.num_workers)