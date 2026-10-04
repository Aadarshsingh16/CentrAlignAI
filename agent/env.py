from abc import ABC, abstractmethod
from typing import Any, Union

class Environment(ABC):
    """
    Abstract environment interface providing ground truth data independent of agent interaction.
    """
    name: str = "environment"

    @abstractmethod
    def ground_truth(self) -> Union[dict[str, Any], list[Any]]:
        """Returns a snapshot of the real system state directly from the underlying data store."""
        pass
