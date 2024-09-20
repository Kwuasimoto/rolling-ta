import importlib.resources as pkg


class DataWriter:
    _file: str

    def __init__(self, file: str = None) -> None:
        self._file = file
        self._resources = pkg.files("resources")
