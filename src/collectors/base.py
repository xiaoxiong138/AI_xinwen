from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseCollector(ABC):
    @abstractmethod
    def collect(self) -> List[Dict[str, Any]]:
        raise NotImplementedError
