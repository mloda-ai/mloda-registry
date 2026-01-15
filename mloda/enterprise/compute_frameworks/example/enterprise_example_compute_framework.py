"""Enterprise Example ComputeFramework implementation."""

from typing import Optional, Set
from uuid import UUID, uuid4

from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode
from mloda.core.abstract_plugins.function_extender import Extender
from mloda.provider import ComputeFramework


class EnterpriseExampleComputeFramework(ComputeFramework):
    """Enterprise Example ComputeFramework for demonstrating plugin structure."""

    def __init__(
        self,
        mode: ParallelizationMode = ParallelizationMode.SYNC,
        children_if_root: frozenset[UUID] = frozenset(),
        uuid: UUID = uuid4(),
        function_extender: Optional[Set[Extender]] = None,
    ) -> None:
        """Initialize with default values for minimal instantiation."""
        super().__init__(mode, children_if_root, uuid, function_extender)
