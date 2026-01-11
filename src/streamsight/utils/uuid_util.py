import datetime
from uuid import NAMESPACE_DNS, UUID, uuid5


def generate_algorithm_uuid(name: str) -> UUID:
    """Generate a UUID for an algorithm based on its name and current timestamp."""
    return uuid5(NAMESPACE_DNS, f"{name}{datetime.datetime.now()}")
