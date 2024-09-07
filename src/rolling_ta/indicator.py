class Indicator:

    _period: int

    def __init__(self, period: int) -> None:
        self._period = period

    def update(self, *args, **kwargs):
        pass

    def calculate(self, *args, **kwargs):
        pass
