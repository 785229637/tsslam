from abc import ABC, abstractmethod
from typing import Generic, Optional, Tuple, TypeVar

from torch import Tensor, nn

from transplat.src.dataset.types import BatchedViews, DataShim
from transplat.src.model.types import Gaussians

T = TypeVar("T")


class Encoder(nn.Module, ABC, Generic[T]):
    cfg: T

    def __init__(self, cfg: T) -> None:
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(
        self,
        context: BatchedViews,
        existing_gaussians: Optional[Gaussians] = None,
        deterministic: bool = False,
    ) -> Tuple[Gaussians, Optional[Tensor], Optional[Tensor]]:
        pass

    def get_data_shim(self) -> DataShim:
        """The default shim doesn't modify the batch."""
        return lambda x: x
