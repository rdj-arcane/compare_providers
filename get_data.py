import os
from enum import StrEnum
from datetime import datetime
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

    def eq_to_pl(self, ts: Timeseries) -> pl.DataFrame:
        df = ts.to_dataframe()
        df.reset_index(inplace=True)

        raw_columns = df.columns.get_level_values(-1)
        columns_list = raw_columns.to_list()
        columns_list = [VALUE_TIME_COL, VALUE_COL]
        df.columns = columns_list
        eq_df = pl.from_pandas(df)

        return eq_df

    def get_eq_single_forecat(
        self, forecast_time: datetime, curve: EQCurves
    ) -> pl.DataFrame:
        df_forecast_ts = self.eq.instances.get(
            curve=curve.value,
            issued=forecast_time,
            tag="ec-ens",
            frequency=Frequency.PT1H,
            ensembles=False,
        )

        df_raw_forecast = self.eq_to_pl(df_forecast_ts)

        df_forecast = (
            df_raw_forecast.with_columns(pl.lit(forecast_time).alias(FORECAST_TIME_COL))
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
            .select(
                FORECAST_TIME_COL + "_utc", FORECAST_TIME_COL, VALUE_TIME_COL, VALUE_COL
            )
        )

        return df_forecast

    def get_eq(self):
        return self.eq


if __name__ == "__main__":
    tz = ZoneInfo("Europe/Copenhagen")
    date_from = datetime(2024, 1, 1, tzinfo=tz)
    date_to = datetime(2024, 9, 1, tzinfo=tz)

    # get_enfor(date_from, date_to)
    # get_actual_production(date_from, date_to)

    load_dotenv()
    api_key = os.getenv("EQ_API_KEY")
    eq = EQ(api_key)
    dt = datetime(2024, 1, 1, 12, 0, 0)
    df_forecast = eq.get_eq_single_forecat(dt, EQCurves.DK_WIND)
    print(df_forecast)
