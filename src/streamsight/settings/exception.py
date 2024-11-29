class EOWSettingError(Exception):
    """End of Window Setting Exception."""

    def __init__(self, message: str | None = None) -> None:
        if not message:
            message = "End of Window reached for the setting."
        self.message = message
        super().__init__(self.message)
