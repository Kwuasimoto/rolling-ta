# Next Steps

## Enhance RollingWindow

in @src/ta/math/rolling.rs, currently there is no respect for if the current candle has closed or is still open. We want to add temporal management that does the following:

THE TEMPORAL MANAGEMENT OF ROLLING WINDOW MUST FOLLOW (DO NOT INFER YOUR OWN SHIT FOR THE LOVE OF GOD):

- Temporal system DOES NOT have its own candle management system. USE the existing @src/ta/math/rolling.rs
- DO NOT add a is_closed variable to the OHLCV struct in @src/ta/types.rs. is_closed bool should be an ephemereal value that does not get saved to memory. It should come from either the ohlcv websocket provider directly, OR be inferred from timestamp information in the OHLCVSeries in the RollingWindow. TO INFER IF THE CANDLE IS CLOSED WITHOUT IS_CLOSED VARIABLE: Take the latest candle out of the RollingWindow and COMPARE ITS TIMESTAMP TO THE TEMPORAL CONFIG (60s, 5m, 15m) etc to determine if is_closed variable is TRUE and whether we should update the latest candle or push a new one to RollingWindow.