"""Base backend from which classes are created for available backends."""
# pylint: disable=unnecessary-pass
# https://docs.python.org/3/library/abc.html uses pass
from abc import ABC, abstractmethod


class BackEnd(ABC):
    """Base class for available backends."""

    @abstractmethod
    def _convert_to_results(self):
        """Convert the backend results to Bambi-readable results."""
        pass

    @abstractmethod
    def build(self):
        """Build the model using the backend."""
        pass

    @abstractmethod
    def run(self):
        """Run the model using the backend."""
        pass
