from datetime import datetime, date
from zoneinfo import ZoneInfo
import polars as pl

import plotly.express as px

from shiny import App, reactive, ui, Inputs, Outputs, Session
from shinywidgets import render_widget, output_widget

ACTUAL_LOAD_COL = "load"
ACTUAL_SOLAR_COL = "solar"
ACTUAL_WIND_COL = "wind"
ACTUAL_WIND_OFFSHORE_COL = "wind_offshore"
ACTUAL_WIND_ONSHORE_COL = "wind_onshore"
BIDDING_ZONE_COL = "bidding_zone"
COR_POWER_COL = "cor_power"
DELIVERY_START_COL = "delivery_start"
FORECAST_TIME_COL = "forecast_time"
FORECAST_TYPE_COL = "forecast_type"
POWER_HOUR_COL = "power_hour"
VALUE_TIME_COL = "value_time"

min_date = date(2024, 1, 1)
max_date = date(2024, 9, 1)

df_enfor = pl.read_parquet("data/enfor.parquet")
df_eq = pl.read_parquet("data/eq.parquet")
df_actuals = pl.read_parquet("data/actuals.parquet").with_columns(
    pl.col(VALUE_TIME_COL).dt.hour().add(1).alias(POWER_HOUR_COL)
)

eq_tags = df_eq.select(pl.col("tag").unique()).get_column("tag").to_list()
eq_tags_sorted = sorted(eq_tags)


def date_to_datetime(date: date) -> datetime:
    return datetime(
        date.year, date.month, date.day, tzinfo=ZoneInfo("Europe/Copenhagen")
    )


app_ui = ui.page_fluid(
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_select("provider", label="Provider", choices=["enfor", "eq"]),
            ui.input_select("tag", label="Tag", choices=eq_tags_sorted),
            ui.input_date_range(
                "date_range", label="Date range", start=min_date, end=max_date
            ),
            ui.input_select(
                "type",
                label="Type",
                choices=["wind", "wind_onshore", "wind_offshore", "solar", "load"],
            ),
            ui.input_numeric(
                "power_hour", label="Power hour", value=0, min=0, max=24, step=1
            ),
        ),
        output_widget("comparison_plot"),
    ),
)


def server(input: Inputs, output: Outputs, session: Session):
    @reactive.Calc
    def get_date_range():
        naive_date_from, naive_date_to = input.date_range()
        date_from = date_to_datetime(naive_date_from)
        date_to = date_to_datetime(naive_date_to)

        return date_from, date_to

    @reactive.Calc
    def get_cols():
        col = input.type()
        actual_col = col + "_actual"
        return col, actual_col

    @reactive.Calc
    def get_power_hour():
        return input.power_hour()

    @reactive.Calc
    def get_forecast_data():
        match input.provider():
            case "enfor":
                df_forecast = df_enfor
            case "eq":
                df_forecast = df_eq.filter(pl.col("tag") == input.tag())

        df = df_forecast.join(df_actuals, on=VALUE_TIME_COL)

        return df

    @reactive.Calc
    def get_data():
        col, actual_col = get_cols()
        date_from, date_to = get_date_range()

        df = get_forecast_data()
        df_relevant = df.filter(
            pl.col(VALUE_TIME_COL).is_between(date_from, date_to),
        )

        power_hour = get_power_hour()
        if power_hour > 0:
            df_relevant = df_relevant.filter(pl.col(POWER_HOUR_COL) == power_hour)

        return df_relevant.select([VALUE_TIME_COL, actual_col, col])

    @reactive.Calc
    def get_limits():
        col, actual_col = get_cols()
        df = get_data()
        x = df.select(pl.min(actual_col).alias("min"), pl.max(actual_col).alias("max"))
        return 0, x[0, "max"]

    @render_widget
    def comparison_plot():
        df = get_data()

        col, actual_col = get_cols()
        fig = px.scatter(df, x=actual_col, y=col, hover_data=[VALUE_TIME_COL])

        x_min, x_max = get_limits()
        fig.add_shape(
            type="line",
            x0=x_min,
            y0=x_min,
            x1=x_max,
            y1=x_max,
            line=dict(color="Red", dash="dash"),
            xref="x",
            yref="y",
        )

        fig.layout.height = 700

        return fig


app = App(app_ui, server)
