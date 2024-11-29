class Registry:
    """A Registry is a wrapper for a dictionary that maps names to Python types.

    Most often, this is used to map names to classes.
    """

    def __init__(self, src) -> None:
        self.registered: dict[str, type] = {}
        self.src = src

    def __getitem__(self, key: str) -> type:
        """Retrieve the type for the given key.

        Args:
            key: The key of the type to fetch.

        Returns:
            The class type associated with the key.
        """
        if key in self.registered:
            return self.registered[key]
        else:
            return getattr(self.src, key)

    def __contains__(self, key: str) -> bool:
        """Check if the given key is known to the registry.

        Args:
            key: The key to check.

        Returns:
            True if the key is known, False otherwise.
        """
        try:
            self[key]
            return True
        except AttributeError:
            return False

    def get(self, key: str) -> type:
        """Retrieve the value for this key.

        This value is a Python type, most often a class.

        Args:
            key: The key to fetch.

        Returns:
            The class type associated with the key.
        """
        return self[key]

    def register(self, key: str, cls: type) -> None:
        """Register a new Python type (most often a class).

        After registration, the key can be used to fetch the Python type from the registry.

        Args:
            key: Key to register the type at. Needs to be unique to the registry.
            cls: Class to register.

        Raises:
            KeyError: If the key is already registered.
        """
        if key in self:
            raise KeyError(f"key {key} already registered")
        self.registered[key] = cls
