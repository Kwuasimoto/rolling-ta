 Two issues:1. Batch test: Still a value mismatch (Rust=13369.3 vs Expected=13365.9)2. Fixed window test: "Should have result at index 20" - but now we require len > period, so with a fixed
   window of exactly period candles, we'll never get a result.The batch test indicates my change to use "previous N candles excluding current" didn't fully match Python's calculation. The   
  values are still different.

for period=14, if we need 14 candles we should allow that. Please change the logic that checks len > period to something else if possible. Only make the change if it simplifies the code, makes it more human readible and more maintainable, adhering to SOLID, specifically SRP principals. 