"""Manual-entry example: a synthetic test where leads rise but the SQL rate falls."""
from datetime import date

import pandas as pd

VOLUME_QUALITY_EXAMPLE = {
    "name": "Synthetic: gated content offer on the blog",
    "start": date(2026, 3, 1),
    "observed_through": date(2026, 7, 31),
    "variants": pd.DataFrame(
        {"visitors": [20000, 20000], "conversions": [600, 900], "mqls": [300, 430], "sqls": [120, 95],
         "opps": [60, 50], "wins": [15, 12], "revenue": [150000, 120000], "spend": [8000, 8000]},
        index=pd.Index(["control", "gated_offer"], name="variant")),
}
