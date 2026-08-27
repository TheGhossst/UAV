import json

import numpy as np

from src.experiments.aodt_compare import encode_uav_xy, expand_positions


def test_encode_and_expand_uav_positions():
    xy = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
    blob = encode_uav_xy(xy, 100.0)
    assert json.loads(blob)[1] == [30.0, 40.0, 100.0]
    rows = [
        {
            "seed": 100,
            "method": "sca",
            "uav_xy": blob,
            "n_uav": 3,
            "n_iot": 10,
            "feasible": 1,
            "sum_rate": 1.0,
        }
    ]
    pos = expand_positions(rows)
    assert len(pos) == 3
    assert pos[2]["uav_id"] == 2
    assert pos[2]["x"] == 50.0
    assert pos[2]["z"] == 100.0
