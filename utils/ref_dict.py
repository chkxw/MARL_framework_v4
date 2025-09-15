class RefDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._refs = {}  # Store reference mappings
        self._key_order = list(super().keys())  # Track insertion order of all keys

    def add_ref(self, ref_key, target_key):
        """Make ref_key always return the value of target_key."""
        self._refs[ref_key] = target_key
        if ref_key not in self._key_order:
            self._key_order.append(ref_key)

    def __getitem__(self, key):
        if key in self._refs:
            # Follow the reference
            return super().__getitem__(self._refs[key])
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        if key in self._refs:
            # Set value for the target key
            super().__setitem__(self._refs[key], value)
        else:
            super().__setitem__(key, value)
            if key not in self._key_order:
                self._key_order.append(key)

    def __contains__(self, key):
        return key in self._refs or super().__contains__(key)

    def __delitem__(self, key):
        if key in self._refs:
            # Remove reference key
            del self._refs[key]
            if key in self._key_order:
                self._key_order.remove(key)
        else:
            # Remove actual key
            super().__delitem__(key)
            if key in self._key_order:
                self._key_order.remove(key)

    def keys(self):
        """Return all keys including reference keys."""
        return [key for key in self._key_order if key in self or key in self._refs]

    def values(self):
        """Return values in insertion order."""
        return [self[key] for key in self.keys()]

    def __iter__(self):
        """Iterate over all keys including reference keys in insertion order."""
        return iter(self.keys())

    def actual_keys(self):
        """Return keys that have actual storage (not references)."""
        return list(super().keys())

    def ref_keys(self):
        """Return keys that are references."""
        return list(self._refs.keys())