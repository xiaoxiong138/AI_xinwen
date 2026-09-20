from abc import ABC, abstractmethod
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List


def parse_feed_entry_date(entry: Any) -> datetime:
    """Parse feed dates without relying on the platform's local-time range."""
    date_was_present = False
    for field in ("published_parsed", "updated_parsed"):
        parsed = getattr(entry, field, None)
        if not parsed:
            continue
        date_was_present = True
        try:
            values = [int(value) for value in parsed[:6]]
            return datetime(*values, tzinfo=timezone.utc)
        except (TypeError, ValueError, OverflowError, OSError):
            continue

    for field in ("published", "updated"):
        raw_value = entry.get(field, "") if hasattr(entry, "get") else getattr(entry, field, "")
        if not str(raw_value or "").strip():
            continue
        date_was_present = True
        try:
            parsed_date = parsedate_to_datetime(str(raw_value))
            if parsed_date.tzinfo is None:
                parsed_date = parsed_date.replace(tzinfo=timezone.utc)
            return parsed_date.astimezone(timezone.utc)
        except (TypeError, ValueError, OverflowError, OSError):
            continue

    if date_was_present:
        return datetime.min.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


class BaseCollector(ABC):
    @abstractmethod
    def collect(self) -> List[Dict[str, Any]]:
        raise NotImplementedError
