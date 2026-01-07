class TimestampAttributeMissingError(Exception):
    """Error raised when timestamp attribute is missing."""

    def __init__(self, message: str = "InteractionMatrix is missing timestamps."):
        super().__init__(message)