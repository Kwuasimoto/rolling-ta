class DataLoader:

    _headers = [
        "Timestamp",
        "timestamp",
        "ts",
        "Close",
        "close",
        "c",
        "Volume",
        "volume",
        "v",
    ]

    def __init__(self) -> None:
        pass

    def read_resource(self, file_name: str):
        pass

    def read_file(self, path: str):
        pass

    def expected_headers(self):
        return self._headers
