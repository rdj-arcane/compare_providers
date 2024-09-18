import re
from pathlib import Path
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
        for col in ["solar", "wind", "wind_offshore", "wind_onshore", "load"]
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


def compute_eq(df_raw_eq: pl.DataFrame) -> pl.DataFrame:
    df_dah = df_raw_eq.with_columns(
        pl.col(FORECAST_TIME_COL).dt.date().alias(FORECAST_DATE_COL),
        pl.col(VALUE_TIME_COL).dt.date().alias(VALUE_DATE_COL),
        pl.col(FORECAST_TIME_COL + "_utc").dt.hour().alias("forecast_hour"),
    ).filter(
        pl.col(VALUE_DATE_COL).sub(pl.col(FORECAST_DATE_COL)) == pl.duration(days=1),
        pl.col("forecast_hour") == 6,
    )

    df_long_eq = (
        df_dah.with_columns(
            pl.col([COMMODITY_COL, LOCATION_COL]).str.to_lowercase(),
        )
        .with_columns(
            pl.col(LOCATION_COL).replace("", None),
        )
        .with_columns(
            pl.concat_str(
                [pl.col(COMMODITY_COL), pl.col(LOCATION_COL)],
                separator="_",
                ignore_nulls=True,
            ).alias("production")
        )
    )

    df_wide_eq = df_long_eq.pivot(
        on=["production"],
        index=[FORECAST_TIME_COL, VALUE_TIME_COL, TAG_COL],
        values=VALUE_COL,
    ).sort([FORECAST_TIME_COL, VALUE_TIME_COL])

    df_eq = (
        df_wide_eq.with_columns(pl.col(VALUE_TIME_COL).dt.truncate("1h"))
        .group_by(FORECAST_TIME_COL, VALUE_TIME_COL, TAG_COL)
        .agg(cs.numeric().mean())
        .sort(FORECAST_TIME_COL, VALUE_TIME_COL)
    )

    return df_eq


def parse_refinitiv_csv(csv_file) -> pl.LazyFrame:
    df = (
        pl.scan_csv(
            csv_file,
            separator="|",
            skip_rows=1,
            schema={
                "Id": pl.Utf8,
                "ForecastDate": pl.Utf8,
                "ValueDate": pl.Utf8,
                "Value": pl.Float32,
            },
        )
        .rename(
            {
                "Id": SERIES_ID_COL,
                "ForecastDate": FORECAST_TIME_COL + "_utc",
                "ValueDate": VALUE_TIME_COL + "_utc",
                "Value": VALUE_COL,
            }
        )
        .with_columns(
            [
                cs.ends_with("_utc").str.to_datetime(
                    "%d.%m.%Y %H:%M:%S", time_unit="ns", time_zone="UTC"
                )
            ]
        )
    )

    return df


def read_refinitiv(csv_dir: Path) -> pl.DataFrame:
    csv_files = list(csv_dir.glob("*.CSV"))
    queries = [parse_refinitiv_csv(csv_file) for csv_file in csv_files]
    query = pl.concat(queries)

    median_ids = pl.LazyFrame(
        {
            SERIES_ID_COL: [
                "106330089",
                "106330238",
                "117637622",
                "117637818",
                "117637902",
                "117637668",
            ],
            PRODUCTION_COL: [
                "solar",
                "solar",
                "wind_onshore",
                "wind_onshore",
                "wind_offshore",
                "wind_offshore",
            ],
            BIDDING_ZONE_COL: ["dk1", "dk2", "dk1", "dk2", "dk1", "dk2"],
        }
    )

    df = query.join(median_ids, on=SERIES_ID_COL, how="inner").collect()

    return df


def compute_refinitiv(df_raw_refinitiv: pl.DataFrame) -> pl.DataFrame:
    df_dah_refinitiv = (
        df_raw_refinitiv.with_columns(
            cs.ends_with("_utc")
            .dt.convert_time_zone("Europe/Copenhagen")
            .name.map(lambda c: c.rstrip("_utc"))
        )
        .with_columns(
            pl.col(FORECAST_TIME_COL).dt.date().alias(FORECAST_DATE_COL),
            pl.col(VALUE_TIME_COL).dt.date().alias(VALUE_DATE_COL),
            pl.col(FORECAST_TIME_COL + "_utc").dt.hour().alias("forecast_hour"),
        )
        .filter(
            pl.col(VALUE_DATE_COL).sub(pl.col(FORECAST_DATE_COL))
            == pl.duration(days=1),
            pl.col("forecast_hour") == 6,
        )
    )

    df_wide_refinitiv = (
        df_dah_refinitiv.pivot(
            on=PRODUCTION_COL,
            index=[FORECAST_TIME_COL, VALUE_TIME_COL, BIDDING_ZONE_COL],
            values=VALUE_COL,
        )
        .with_columns(
            pl.fold(0, lambda acc, s: acc + s, cs.starts_with("wind")).alias("wind"),
        )
        .with_columns(
            cs.numeric().round(2).name.keep(),
        )
        .sort(FORECAST_TIME_COL, VALUE_TIME_COL)
    )

    return df_wide_refinitiv


def compute_meteologica(df_raw_meteologica: pl.DataFrame) -> pl.DataFrame:
    df_meteologica = df_raw_meteologica.with_columns(
        pl.col(VALUE_TIME_COL).dt.cast_time_unit("ns"),
    )

    return df_meteologica


if __name__ == "__main__":
    df_raw_enfor = pl.read_parquet("data/raw_enfor.parquet")
    df_enfor = compute_enfor(df_raw_enfor)
    df_enfor.write_parquet("data/enfor.parquet")

    df_raw_actuals = pl.read_parquet("data/raw_actuals.parquet")
    df_actuals = compute_actuals(df_raw_actuals)
    df_actuals.write_parquet("data/actuals.parquet")

    df_raw_eq = pl.read_parquet("data/eq/raw/*.parquet")
    df_eq = compute_eq(df_raw_eq)
    df_eq.write_parquet("data/eq.parquet")

    df_raw_refinitiv = read_refinitiv(Path("data/refinitiv/raw"))
    df_refinitiv = compute_refinitiv(df_raw_refinitiv)
    df_refinitiv.write_parquet("data/refinitiv.parquet")

    df_raw_meteologica = pl.read_parquet("data/raw_meteologica.parquet")
    df_meteologica = compute_meteologica(df_raw_meteologica)
    df_meteologica.write_parquet("data/meteologica.parquet")
