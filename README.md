# Rolling-TA

| Depedencies |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Required    | ![Python Version](https://img.shields.io/badge/python-%3E%3D%203.8-blue) <div style="border-left: 1px solid; height: 20px; margin: 0 10px; margin-top: 5px; display: inline-block;"></div> ![Pandas Version](https://img.shields.io/badge/pandas-%3E%3D%202.2.2-blue) <div style="border-left: 1px solid; height: 20px; margin: 0 10px; display: inline-block;"></div> ![Numba Version](https://img.shields.io/badge/numba-%3E%3D%200.60.0-blue) <div style="border-left: 1px solid; height: 20px; margin: 0 10px; margin-top:8px; display: inline-block;"></div> ![Dotenv Version](https://img.shields.io/badge/python_dotenv-%3E%3D%201.0.1-blue) |
| Optional    | ![Openpyxl Version](https://img.shields.io/badge/openpyxl-%3E%3D%203.1.5-blue)<div style="height: 20px; margin: 0 10px; margin-top: 5px; display: inline-block;"></div>                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |

| Tests  |                                                                                                                                                                                                                                                          |
| ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| pytest | ![Coverage](https://img.shields.io/badge/coverage-98%25-brightgreen)<div style="border-left: 1px solid; height: 20px; margin: 0 10px; margin-top: 8px; display: inline-block;"></div>![Passing](https://img.shields.io/badge/passing-100%25-brightgreen) |

### **Primary Errors (to be fixed at some point)**

**Does not support intra timestep updates**: </br>
The update functions for the indicators are not intelligent _yet_, they do not know if the new data being processed is a terminated candle. If you're working with a 1 minute timeframe, and you update SMA (example) 5 times within a minute, the last 5 values of the SMA will be within that TimeFrame. This will have to be handled on your end until I'm requested to, or deem it necessary to implement this into the code. Happy Trading 🤓

### Update Schedule

**Not being worked on**: </br>
I am currently working on some ML stuff for the foreseeable future, this library will likely get some more love in 1-2 months once I've gotten other private projects up to speed.

### 0.8.2 -> 0.8.3

- Use to\_\* functions to get indicator data

## TOC

- [Project Description](#project-description)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Basic Example](#basic-example)
- [Configuration](#environment)
- [Tests](#tests)
- [Todo](#todo)
- [Contributing](#contributing)
- [License](#license)

## Project Description

Leveraing **Numpy** and **Numba**, rolling-ta is designed for fast and efficient technical analysis, while maintaining the simplicity and ease of use of **Pandas**. It provides an intuitive single-responsibility API focused on technical analysis calculations and real-time updates. Prioritizing optimizations for speed, while being lightweight.

Key features include:

    - Fast computations by leveraging the best features of NumPy and Numba.
    - Incremental updates support, making it suitable for real-time applications.
    - Simple, intuitive API for usage/extension without sacrificing performance.

Whether you're calculating simple indicators like SMA, or advanced indicators leveraging Linear Regression, this library delivers powerful performance with a clean and minimalistic interface.

By default, a lot of the stuff like **ignore gil** and **parallel** are set to false to increase compatibility. Please view the _(Recommended)_ [environment](#environment) variables to make the library even faster.

## Getting Started

### Installation

```bash
py -m pip install rolling-ta
```

### Basic Example

**Initilization**

```py
from rolling_ta.trend import SMA

# With pyopenxl (see github repo for more info)
from rolling_ta.data import CSVLoader

loader = CSVLoader()

# Load a csv, headers not support (use columns param)
# default columns=[timestamp, open, high, low, close, volume]
data = loader.read_resources()

# returns unnamed pd.Series
sma = SMA(data).to_numpy(dtype=np.float64)
```

**Rolling updates**

```py
sma.update(data.iloc[index])
```

**Getting indicator as different types**

```py
sma_raw = sma.to_array()
sma_np = sma.to_numpy()
sma_pd = sma.to_series()
```

**Getting an imbedded indicator**

```py
adx = ADX(data)

positive_directional_movement = adx.to_numpy(get="pdmi", dtype=np.float64)
```

## Environment

rolling-ta uses **python-dotenv** to read environment variables.

_(Recommended)_
</br>Tell **Numba** to write compiled machine code to \_\_pycache\_\_.
</br>If set to 1, Numba will attempt to read compiled machine code into RAM using pickle.
</br>This speeds up subsequent program runs.

```dotenv
NUMBA_DISK_CACHING=0|1
```

_(Recommended w/ NUMBA_DISK_CACHING=1)_
</br>Tell **Numba** to aggressively compile mathematical operations.
</br> Attempts to increase efficiency, slightly increases compile time.

```dotenv
NUMBA_FASTMATH=0|1
```

_(Recommended)_
</br>Run looping operations (nb.prange) in parallel.
</br>Drastically speeds up computations with arrays.size >= 100_000
</br>See [Linear Regression Indicator](./tests/speed/lri.ipynb) speed tests.

```dotenv
NUMBA_PARALLEL=0|1
```

Tell **Numba** functions to ignore the python global interpreter lock.
</br> The speed increase with this set to true is 5-10% +/- 1%.

```dotenv
NUMBA_NOGIL=0|1
```

For more information on **Numba** compilation, see: </br>
[Numba compilation options](https://numba.readthedocs.io/en/stable/user/jit.html#compilation-options)

## Tests

### Speed Tests

When comparing against [ta](https://github.com/bukosabino/ta), a popular financial library using purely **Pandas** and **Numpy**, we see some significant performance increases across some indicators.

_see_ [speed tests](./tests/speed)

| Indicator                  | Lib               | Speed                                                    |
| -------------------------- | ----------------- | -------------------------------------------------------- |
| Average Directional Index  | Rolling-TA</br>TA | 813 μs ± 102 μs per loop</br>93.7 ms ± 544 μs per loop   |
| Average True Range         | Rolling-TA</br>TA | 1.32 ms ± 118 μs per loop</br>40.5 ms ± 835 μs per loop  |
| Exponential Moving Average | Rolling-TA</br>TA | 52.8 μs ± 1.17 μs per loop</br>138 μs ± 4.43 μs per loop |
| Ichimoku Cloud             | Rolling-TA</br>TA | 216 μs ± 65.8 μs per loop</br>3.07 ms ± 316 μs per loop  |
| Money Flow Index           | Rolling-TA</br>TA | 382 μs ± 11.2 μs per loop</br>122 ms ± 1.92 ms per loop  |
| On Balance Volume          | Rolling-TA</br>TA | 51.5 μs ± 819 ns per loop</br>272 μs ± 8.19 μs per loop  |
| Relative Strength Index    | Rolling-TA</br>TA | 172 μs ± 100 μs per loop</br>1.06 ms ± 49.7 μs per loop  |
| Simple Moving Average      | Rolling-TA</br>TA | 47.7 μs ± 649 ns per loop</br>264 μs ± 14.4 μs per loop  |
| True Range                 | Rolling-TA</br>TA | 72.5 μs ± 1.14 μs per loop</br>2 ms ± 110 μs per loop    |

### Truth Tests

These tests confirm that the results from python implementations match excel implementations.

Tests [pytests](./tests/indicators/)
</br> Fixtures [fixtures](./tests/fixtures/data_sheets.py)
</br> Excel Sheets [resources](./resources/)

## Todo

_Expected Updates for 0.9_

- Create update fn overloaded signature that accepts just the price information, instead of forcing people to supply a pd.Series object.
- Allow users to pass a mutable array to Indicator.update() function, instead of having to rebuild their existing data.
  - Potential implementation (might be in 1.0), intelligently read existing data and parse required information from it to perform incremental updates.
- Implement Indicator.to_numpy(), Indicator.to_series() for more explicit data return.
- Finish update functions for the following Indicators:
  - Linear Regression Indicator (LRI)
  - Linear Regression Forecast (LRF)
  - Linear Regression R2 (lr2)
- Maybe migrate pandas to polars and test GPU vectoring speed?

## Contributing

If you wish to provide a contribution, please follow these steps:

- Fork the dev branch.
- Implement your changes.
- Run ./pytest to confirm nothing breaks
  </br>_(You'll need the excel resources from above. Or your own with a link so I can confirm its accuracy)_
- Run the new implementations within [Speed Tests](./tests/speed/) in a python notebook cell affixed with %%timeit.

If you don't follow these steps, the contribution will not be considered.

## License

[License](./LICENSE)
