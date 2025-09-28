# Maritime Route Prediction Toolkit

This repository contains utilities for turning a dense historical AIS-like
track library into a reusable road-network style representation and for
predicting future legs of a vessel by matching its recent track against the
historical library.

## Features

- **Route network extraction** – Merges nearby historical points into reusable
  nodes and edges so multiple vessels travelling along the same sea lane share a
  single route corridor.
- **Historical track templating** – Stores per-track node sequences, enabling the
  predictor to match partial observations against real traversals.
- **Destination aware prediction** – Checks whether the user-specified
  destination lies along the matched historical route before reusing it.
- **Land mask support** – Optional GeoJSON polygons can be rendered on the
  visualisation to provide context and highlight land avoidance.
- **Front-end friendly output** – The prediction result includes a natural
  language explanation, the supporting historical track information, the
  predicted track coordinates and a list of ship types that the model can
  currently support according to the historical library.

## Getting Started

1. Install dependencies (a lightweight scientific Python stack is sufficient):

   ```bash
   pip install -r requirements.txt
   ```

2. Build the network and run a prediction from the command line:

   ```bash
   python predict_route_cli.py \
       --history scene_coast_patrol.csv \
       --observed sample_observed.csv \
       --destination 23.75 132.70 \
       --output prediction.png \
       --land-mask "land_mask (6).geojson"
   ```

   The command prints a JSON report and generates a `prediction.png` map.

## Integrating into Your System

- Use :func:`route_prediction.load_historical_tracks` to load the historical
  dataset once during service initialisation.
- Create a :class:`route_prediction.RouteNetworkBuilder` to build the network and
  retain the resulting :class:`route_prediction.RouteNetwork` for future
  requests.
- For each incoming prediction request convert the provided observations into a
  list of :class:`route_prediction.TrackPoint` objects and call
  :class:`route_prediction.RoutePredictor`.
- When a suitable historical route is found, the predictor returns a detailed
  explanation and future track segments that can be fed directly to the front
  end. When the data support is insufficient the response clearly communicates
  the limitation.

### Working with In-Memory Python Data

If your application already has the tracks loaded in memory (e.g. from a
database query or an API) you can skip the CSV step entirely and use the
``*_from_records`` helpers:

```python
from route_prediction import predict_route_from_records

history_rows = [
    {"ph": "890012", "lon": 133.0352778, "lat": 23.7838889, "mbc": 262.6, "mbv": 15.6, "time": "2020-10-31 09:23:12+08:00"},
    # ... more history rows ...
]

observed_rows = [
    {"lon": 133.10, "lat": 23.75, "mbc": 260.0, "mbv": 16.0, "time": "2023-02-10 10:10:00+08:00"},
    # ... most recent observations ...
]

result = predict_route_from_records(
    history_rows,
    observed_rows,
    destination=(23.75, 132.70),
    history_defaults={"mbmc": "未知舰艇", "gjdq": "未知", "type": "未知类型"},
)
```

When your data uses different field names provide ``history_aliases`` or
``observed_aliases`` to map them into the expected schema. For example, if your
history uses ``rpLong`` and ``h`` columns you can call::

    predict_route_from_records(history_rows, observed_rows, destination,
                                history_aliases={"rpLong": "lon", "h": "mbc"})

The helper returns the same dictionary structure as the CLI, making it easy to
integrate into web services.

## Visualising Predictions Programmatically

The helper :func:`route_prediction.save_prediction_plot` function accepts the
observed points, the predicted continuation and optionally a subset of historical
tracks (e.g. the matched one) plus an optional land mask. It produces a PNG map
that can be sent to stakeholders or stored for audits.

## Supported Ship Types

The predictor automatically reports all ship types discovered in the historical
library. This allows operators to quickly determine whether the current target
falls within the modelled fleet or should be flagged as unsupported.

## Data Privacy

The toolkit treats the historical data as immutable input and does not store any
additional personally identifiable information. Ensure your deployment continues
complying with relevant regulations when integrating with operational systems.
