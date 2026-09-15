# Data Model: Demo Normalize Flow

## Environment State

- `VITE_TAHIMIK_DATA_MODE`: `live` enables API inference; all other/missing values select demo mode.

## Demo State Transaction

`DEMO_SENTENCE` is copied into `inputText` and `batchRows`; `isBatch=false`, `selected=0`, `showSummary=false`, and `apiStatusMessage=null` are applied together.
