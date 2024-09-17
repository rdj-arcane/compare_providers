import os
from enum import StrEnum
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
import pandas as pd
import polars as pl

from energyquantified import EnergyQuantified
from energyquantified.time import Frequency
from energyquantified.data.timeseries import Timeseries
from dotenv import load_dotenv

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


class EQCurves(StrEnum):
    DK_WIND = "DK Wind Power Production MWh/h 15min Forecast"
    DK_WIND_OFFSHORE = "DK Wind Power Production Offshore MWh/h 15min Forecast"
    DK_WIND_ONSHORE = "DK Wind Power Production Onshore MWh/h 15min Forecast"
    DK_SOLAR = "DK Solar Photovoltaic Production MWh/h 15min Forecast"


class EQ:
    def __init__(self, api_key):
        self.eq = EnergyQuantified(api_key=api_key)

    def single_ts_to_pl(self, ts: Timeseries) -> pl.DataFrame:
        df_pd = ts.to_dataframe()
        df_pd.reset_index(inplace=True)
        raw_df = pl.from_pandas(df_pd)
        raw_df.columns = [VALUE_TIME_COL, VALUE_COL]

        forecast_time = ts.instance.issued
        tag = ts.instance.tag
        categories = ts.curve.categories
        if "Wind" in categories:
            commodity = "Wind"
        elif "Solar" in categories:
            commodity = "Solar"
        else:
            commodity = ""

        if "Offshore" in categories:
            location = "Offshore"
        elif "Onshore" in categories:
            location = "Onshore"
        else:
            location = ""

        df = (
            raw_df.with_columns(
                pl.lit(forecast_time).alias(FORECAST_TIME_COL),
                pl.lit(commodity).alias(COMMODITY_COL),
                pl.lit(location).alias(LOCATION_COL),
                pl.lit(tag).alias(TAG_COL),
            )
            .with_columns(
                pl.col([VALUE_TIME_COL, FORECAST_TIME_COL])
                .dt.convert_time_zone("Europe/Copenhagen")
                .dt.cast_time_unit("ns")
            )
            .with_columns(
                pl.col(FORECAST_TIME_COL)
                .dt.convert_time_zone("UTC")
                .alias(FORECAST_TIME_COL + "_utc")
            )
        )

        return df

    def get_eq_single_forecat(
        self, forecast_time: datetime, curve: EQCurves
    ) -> pl.DataFrame:
        df_forecast_ts = self.eq.instances.load(
            curve=curve.value, issued_at_latest=forecast_time, limit=10
        )

        df_forecast_list = [self.single_ts_to_pl(ts) for ts in df_forecast_ts]

        df_forecast = pl.concat(df_forecast_list)

        return df_forecast

    def get_eq(self, date_from: datetime, date_to: datetime) -> None:
        save_dir = Path("data/eq/raw")
        if not save_dir.exists():
            save_dir.mkdir(parents=True)

        for curve in list(EQCurves):
            print(f"Getting {curve.name}")
            current_date = date_from
            while current_date < date_to:
                current_date += timedelta(days=1)
                save_file = (
                    save_dir
                    / f"{curve.name}_{current_date.isoformat().replace(':', '_')}.parquet"
                )
                if save_file.exists():
                    continue

                df = self.get_eq_single_forecat(current_date, curve)
                df.write_parquet(save_file)


if __name__ == "__main__":
    tz = ZoneInfo("Europe/Copenhagen")
    date_from = datetime(2024, 1, 1, tzinfo=tz)
    date_to = datetime(2024, 9, 1, tzinfo=tz)

    # get_enfor(date_from, date_to)
    # get_actual_production(date_from, date_to)

    load_dotenv()
    api_key = os.getenv("EQ_API_KEY")
    eq = EQ(api_key)
    dt = date_from + timedelta(hours=11, minutes=10)
    eq.get_eq(dt, date_to)
