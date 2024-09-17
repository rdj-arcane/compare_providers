import re
import polars as pl
import polars.selectors as cs

from dbquery.nordpool.actuals import actuals_production_extract_latest

from constants import *


def map_asset_location(input_str: str) -> str:
    match = re.match(r'{"(.*?)","(.*?)"}', input_str)
    if match:
        asset, location = match.groups()
        return f"{asset}_{location}"
    else:
        return input_str


def compute_enfor(df_raw_enfor: pl.DataFrame) -> pl.DataFrame:
    df_dah_enfor = df_raw_enfor.with_columns(
        pl.col(FORECAST_TIME_COL).dt.date().alias(FORECAST_DATE_COL),
        pl.col(VALUE_TIME_COL).dt.date().alias(VALUE_DATE_COL),
        pl.col(FORECAST_TIME_COL).dt.hour().alias("forecast_hour"),
    ).filter(
        pl.col(VALUE_DATE_COL).sub(pl.col(FORECAST_DATE_COL)) == pl.duration(days=1),
        pl.col("forecast_hour") == 11,
    )

    df_long_enfor = (
        df_dah_enfor.with_columns(
            pl.col(ASSET_KEY_COL)
            .str.split("_")
            .list.get(1, null_on_oob=True)
            .alias("location"),
            pl.col(ASSET_KEY_COL)
            .str.extract(r"([a-z]+[1-9]?)_?")
            .alias(BIDDING_ZONE_COL),
        )
        .with_columns(
            pl.col("location")
            .replace({"land": "onshore", "sea": "offshore"})
            .name.keep()
        )
        .with_columns(
            pl.concat_str(
                [pl.col(FORECAST_TYPE_COL), pl.col("location")],
                separator="_",
                ignore_nulls=True,
            ).alias("production")
        )
    )

    df_wide_enfor = (
        df_long_enfor.pivot(
            on=["production"],
            index=[FORECAST_TIME_COL, VALUE_TIME_COL, BIDDING_ZONE_COL],
            values=COR_POWER_COL,
        )
        .rename(map_asset_location)
        .with_columns(
            pl.fold(0, lambda acc, s: acc + s, cs.starts_with("wind")).alias("wind"),
        )
        .sort([FORECAST_TIME_COL, VALUE_TIME_COL])
    )

    return df_wide_enfor


def compute_actuals(df_raw_actuals: pl.DataFrame) -> pl.DataFrame:
    rename_map = {
        col: col + "_actual"
        for col in ["solar", "wind", "wind_offshore", "wind_onshore"]
    }
    rename_map[DELIVERY_START_COL] = VALUE_TIME_COL

    df_latest_actuals = actuals_production_extract_latest(df_raw_actuals)

    df_actuals = (
        df_latest_actuals.with_columns(
            pl.col(BIDDING_ZONE_COL).str.to_lowercase(),
            pl.fold(0, lambda acc, s: acc + s, cs.starts_with("wind")).alias("wind"),
        )
        .rename(rename_map)
        .select(VALUE_TIME_COL, BIDDING_ZONE_COL, cs.ends_with("_actual"))
        .sort(VALUE_TIME_COL)
    )

    return df_actuals


if __name__ == "__main__":
    df_raw_enfor = pl.read_parquet("data/raw_enfor.parquet")
    df_enfor = compute_enfor(df_raw_enfor)
    df_enfor.write_parquet("data/enfor.parquet")

    df_raw_actuals = pl.read_parquet("data/raw_actuals.parquet")
    df_actuals = compute_actuals(df_raw_actuals)
    df_actuals.write_parquet("data/actuals.parquet")
