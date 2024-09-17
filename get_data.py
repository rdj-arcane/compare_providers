from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import polars as pl

from dbquery.cache.cache_global import Cache
from dbquery.nordpool.actuals import actuals_production_get

from constants import *

cache = Cache()

save_dir = Path("data")
if not save_dir.exists():
    save_dir.mkdir()


def get_enfor(date_from, date_to):
    df_enfor = cache.fundamentals_get(
        date_from,
        date_to,
        asset_keys=["dk1", "dk1_land", "dk1_sea"],
        forecast_hours=[11],
    )
    enfor_file = save_dir / "raw_enfor.parquet"

    df_enfor.write_parquet(enfor_file)


def get_actual_production(date_from, date_to):
    df_raw_actuals = actuals_production_get(date_from, date_to, ["DK1"])
    actuals_file = save_dir / "raw_actuals.parquet"

    df_raw_actuals.write_parquet(actuals_file)


if __name__ == "__main__":
    tz = ZoneInfo("Europe/Copenhagen")
    date_from = datetime(2024, 1, 1, tzinfo=tz)
    date_to = datetime(2024, 9, 1, tzinfo=tz)

    # get_enfor(date_from, date_to)
    get_actual_production(date_from, date_to)
