class TAException(Exception):
    indicator: str

    def __str__(self):
        return f"\nTAException:\n - type:\t\t Base TA Exception\n - indicator:\t{self.indicator.__class__.__name__}\n - exc:\t\t{self}"


class IndicatorException(TAException):
    def __str__(self):
        return f"\IndicatorException:\n - type:\t\t Base Indicator Exception\n - indicator:\t{self.indicator.__class__.__name__}\n - exc:\t\t{self}"


class IndicatorRollException(TAException):
    def __str__(self):
        return f"\IndicatorRollException:\n - type:\t\t Indicator Roll Exception\n - indicator:\t{self.indicator.__class__.__name__}\n - exc:\t\t{self}"
