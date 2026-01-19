"""Base classes for preprocessing transforms."""

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Callable, Any


T = TypeVar('T')
U = TypeVar('U')


class Transform(ABC, Generic[T, U]):
    """Abstract base class for data transforms.

    Transforms are composable functions that process data items.
    They follow the design pattern used in torchvision.transforms.
    """

    @abstractmethod
    def __call__(self, item: T) -> U:
        """Apply the transform to an item.

        Args:
            item: Input item to transform.

        Returns:
            Transformed item.
        """
        pass


class Compose:
    """Composes several transforms together.

    This is similar to torchvision.transforms.Compose.
    """

    def __init__(self, transforms: list[Any]) -> None:
        """Initialize the composition.

        Args:
            transforms: List of transforms to compose.
        """
        self.transforms = transforms

    def __call__(self, item: Any) -> Any:
        """Apply transforms sequentially.

        Args:
            item: Input item to transform.

        Returns:
            Transformed item after applying all transforms.
        """
        result = item
        for transform in self.transforms:
            result = transform(result)
        return result