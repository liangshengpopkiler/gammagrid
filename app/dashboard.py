"""Streamlit UI. Display and user input only — all business logic lives in
metrics.py (calculations) and collector.py (data collection)."""

from __future__ import annotations

import datetime as dt
import json

import altair as alt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from matplotlib.colors import LinearSegmentedColormap

from app import (  # noqa: I001 — grouped by what they are, not alphabetised
    catalogue,
    collector,
    config,
    day_summary,
    i18n,
    db,
    freshness,
    market_calendar,
    metrics,
    providers,
    suggestions,
    theme,
    weather,
)
from app.viewtime import (
    format_date,
    format_datetime,
    strikes_around_money,
    timezone_label,
    to_viewer,
    with_viewer_index,
)

# Same 2x2 diagonal "chip" mark used as the favicon on gammagrid.io — keep
# this data URI in sync with the teaser site's <link rel="icon"> if the mark
# ever changes.
FAVICON = (
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E"
    "%3Crect width='32' height='32' fill='%230A0C0B'/%3E"
    "%3Crect x='4' y='4' width='10' height='10' fill='%23B833E0'/%3E"
    "%3Crect x='18' y='4' width='10' height='10' fill='%23222'/%3E"
    "%3Crect x='4' y='18' width='10' height='10' fill='%23222'/%3E"
    "%3Crect x='18' y='18' width='10' height='10' fill='%2322C55E'/%3E%3C/svg%3E"
)

st.set_page_config(page_title="GammaGrid", page_icon=FAVICON, layout="wide")

st.markdown(theme.stylesheet(), unsafe_allow_html=True)


def _drop_stale_choice(key: str, options) -> None:
    """Forgets a stored selection that is no longer offered.

    Streamlit raises when a widget's remembered value is absent from its
    options, and these lists narrow as the user drills down — pick an expiry
    that lists strike 275, then one that does not, and the Contract tab would
    die on a perfectly ordinary sequence of clicks."""
    if key in st.session_state and st.session_state[key] not in list(options):
        del st.session_state[key]


# The catalogue, cached. It is ~5,300 rows and the file behind it is downloaded
# weekly, so re-reading it on every rerun would be a query per click for an
# answer that cannot have moved.
#
# Five minutes rather than an hour, and the reason is not the file: the ORDER
# depends on what this installation watches, so a symbol just added should reach
# the top of the list within a rerun or two rather than at the top of the hour.
#
# Caching does not reduce what crosses the wire — the whole list goes to the
# browser on every rerun — and that is the price of filtering without a round
# trip. It is also why this list is not made longer than it has to be.
@st.cache_data(ttl=300)
def _cached_catalogue_options(_conn, watched: tuple[str, ...]) -> list[tuple[str, str]]:
    return catalogue.options(_conn, list(watched))


@st.cache_data(ttl=1800)
def _cached_has_options(ticker: str) -> bool | None:
    """Does this symbol have a chain to collect — asked once per symbol.

    Cached because it is a network call sitting behind a button somebody may
    press twice, and because the answer for a given symbol does not change
    within half an hour. `None` is cached too, and deliberately: when the
    source could not answer, the ticker is accepted, so there is nothing to
    re-ask.

    Called through getattr rather than as a plain method — a provider written
    against the published interface need not have this, and a missing method
    means "cannot find out", which is the same as None.
    """
    check = getattr(providers.get_provider(), "underlying_has_options", None)
    return None if check is None else check(ticker)


@st.cache_data(ttl=1800)
def _cached_collection_depth() -> dict[str, dt.date]:
    """First collection date per ticker. Half an hour is right for a fact that
    only changes when a ticker is collected for the very first time."""
    return db.collection_depth(db.get_connection())




def render_option_detail(
    conn, df: pd.DataFrame, selected_ticker: str, tracked: pd.DataFrame,
    opt_expiry: pd.Timestamp, opt_strike: float, opt_type: str, key_prefix: str,
) -> None:
    """Price/IV/greeks history for a specific contract — shared renderer for the
    Contract tab and for the contract card on the Screener (row click), so the
    logic isn't duplicated in two places. `key_prefix` — tabs render in the same
    script run simultaneously (Streamlit doesn't lazy-render tabs), so widgets
    with identical labels in both places need distinct keys."""
    already_tracked = not tracked.empty and (
        (tracked["expiry"] == opt_expiry)
        & (tracked["strike"] == opt_strike)
        & (tracked["option_type"] == opt_type)
    ).any()
    if already_tracked:
        st.caption(i18n.t("📌 pinned", "📌 已固定"))
    elif st.button(i18n.t("📌 Pin", "📌 固定"), key=f"{key_prefix}_pin_contract"):
        db.add_tracked_contract(conn, selected_ticker, opt_expiry, opt_strike, opt_type)
        st.rerun()

    greeks_history = metrics.contract_greeks_history(df, opt_strike, opt_expiry, opt_type)

    if greeks_history.empty:
        st.info(i18n.t("No history for the selected contract.", "所选合约没有历史数据。"))
        return

    price_column, attribution_column = st.columns(2)
    with attribution_column:
        render_greek_attribution(greeks_history)

    with price_column:
        st.subheader(i18n.t("Option price", "期权价格"))
        st.line_chart(with_viewer_index(greeks_history)["last_price"], color=BRAND_PURPLE)
    with st.expander(i18n.t("ℹ️ How to read this", "ℹ️ 怎么读这张图")):
        st.write(
            i18n.t(
                "Price history of this specific contract (last price at each collection). "
                "Compare the option's price action with the underlying's move and with the IV "
                "dynamics below — if the option is gaining faster than the stock move explains, "
                "it's likely volatility itself that's rising (see Vega in the greeks block).",
                "该合约自身的历史价格(每次采集的最新价)。把期权价格走势与标的涨跌、以及下方的 "
                "IV 动态对照 — 如果期权涨得比标的涨跌能解释的更快,那多半是波动率本身在上升"
                "(见 Greeks 区的 Vega)。",
            )
        )

    st.subheader(i18n.t("Contract implied volatility", "合约隐含波动率"))
    st.line_chart(with_viewer_index(greeks_history)["implied_volatility"], color=BRAND_GREEN)

    latest_iv = greeks_history.iloc[-1]["implied_volatility"]
    # READ, NOT FETCHED. These figures come from daily closes, which cannot
    # change during a session, so they are stored once a day by the collection
    # pass and read here. This used to be a six-month price-history download
    # made while the page was being drawn.
    stored_rv = db.get_realized_volatility(conn, selected_ticker)
    if stored_rv:
        as_of, rv, rv_source = stored_rv
        st.caption(
            i18n.t(
                f"Current contract IV: **{latest_iv:.1%}**. Realized volatility of the underlying "
                f"(close-to-close, from {rv_source} daily closes as of {format_date(as_of)} — its "
                "depth doesn't depend on how many days we've been collecting option chains, which "
                "is why it isn't drawn as a line on the same chart: the time scales are too "
                "different): ",
                f"当前合约 IV:**{latest_iv:.1%}**。标的已实现波动率"
                f"(收盘对收盘,来自 {rv_source} 的日线,截至 {format_date(as_of)};其深浅与期权链"
                "采集天数无关,所以不与本图同框绘制:两者时间尺度差太大):",
            )
            + " · ".join(f"RV({window}d) **{value:.1%}**" for window, value in sorted(rv.items()))
        )
        st.caption(
            i18n.t(
                "IV well above RV — the market is pricing a premium for future uncertainty "
                "(normal ahead of earnings/events). IV below RV is a rare situation — the option "
                "may be underpriced relative to the underlying's actual recent volatility.",
                "IV 显著高于 RV — 市场在为未来的不确定性定价(财报/事件前属正常)。"
                "IV 低于 RV 较为罕见 — 说明期权相对标的近期实际波动可能被低估。",
            )
        )
    else:
        st.caption(
            i18n.t(
                "Realized volatility of the underlying: not stored yet — it is written once a day "
                "by the collection pass, so it appears after the next collection.",
                "标的已实现波动率:尚未存储 — 它由采集流程每天写入一次,下一轮采集后就会出现。",
            )
        )

    st.subheader(i18n.t("Greeks over time", "Greeks 时间序列"))
    greek_cols = ["delta", "gamma", "theta", "vega", "rho", "vanna", "charm"]
    left, right = st.columns(2)
    for i, greek in enumerate(greek_cols):
        target = left if i % 2 == 0 else right
        target.caption(greek.capitalize())
        target.line_chart(
            with_viewer_index(greeks_history)[greek],
            color=BRAND_PURPLE if i % 2 == 0 else BRAND_GREEN,
        )

    with st.expander(i18n.t("📊 Interpreting current values and their trend",
                            "📊 当前数值与趋势解读")):
        for note in metrics.interpret_greeks(greeks_history):
            st.write(f"- {note}")


def render_greek_attribution(greeks_history: pd.DataFrame) -> None:
    """Where the contract's money went, split by greek.

    A WATERFALL, because it is the only shape that shows the arithmetic: the
    bars start at the price the contract had, each greek moves it, and the last
    bar lands on the price it has now — so the residual is not a number beside
    the chart, it is the bar that closes the gap.

    In cents. The terms on a 19c option are hundredths of a dollar, and
    "-0.476" invites a misread that "-47.6c" does not.
    """
    attribution = metrics.greek_attribution(greeks_history)
    st.subheader(i18n.t("Where the price went", "价格去向分解"))
    if attribution.by_day.empty:
        # SAY WHICH REASON IT WAS. There are two, and until this branch
        # separated them the panel reported the wrong one: a contract whose
        # every day sits under the cent floor was told its implied volatility
        # had failed the data-quality guard, which sends the reader hunting for
        # a data problem that does not exist. Measured on the 60 most-collected
        # SPY contracts in a real database: 47 decompose, 13 are refused, and
        # every one of those 13 is refused for being cheap rather than for
        # anything to do with IV.
        if attribution.dropped_cheap:
            st.info(
                i18n.t(
                    f"This contract traded under {config.ATTRIBUTION_MIN_PRICE * 100:.0f}c on "
                    f"{attribution.dropped_cheap} of the day(s) collected, and the split is not "
                    "shown for those: at a few cents a contract, every greek is arithmetic about "
                    "rounding and the decomposition would be noise reported as a finding. Deeper "
                    "in-the-money contracts on this ticker will show it.",
                    f"该合约有 {attribution.dropped_cheap} 个采集日价格低于 "
                    f"{config.ATTRIBUTION_MIN_PRICE * 100:.0f} 美分,这些日子的分解不显示:"
                    "几美分的合约上,每个 Greek 都只是舍入运算,分解结果是纯噪声。"
                    "该代码更深入的实值合约可以看到有效分解。",
                )
            )
        else:
            st.info(
                i18n.t(
                    "Not enough usable days yet to split the price change by greek. This needs "
                    "at least two days whose implied volatility passed the data-quality guard — "
                    "and with collection on a schedule, days that are actually a day apart.",
                    "可用交易日还不足,无法按 Greek 分解价格变化。这需要至少两个 IV 通过"
                    "数据质量校验的日子 — 且在定时采集下,是真正相隔一个交易日的两天。",
                )
            )
        return

    totals = attribution.totals
    labels = [
        i18n.t(en, zh)
        for en, zh in zip(
            ["Start", "Delta", "Gamma", "Vega", "Theta", "Residual", "End"],
            ["期初", "Delta", "Gamma", "Vega", "Theta", "残差", "期末"],
        )
    ]
    steps = [totals[name] * 100 for name in (*metrics.ATTRIBUTION_TERMS, "residual")]
    figure = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute", *["relative"] * 5, "total"],
        x=labels,
        y=[attribution.start_price * 100, *steps, 0],
        text=waterfall_labels(attribution.start_price * 100, steps),
        textposition="outside",
        decreasing={"marker": {"color": BRAND_PURPLE}},
        increasing={"marker": {"color": BRAND_GREEN}},
        totals={"marker": {"color": "#8B8194"}},
        connector={"line": {"color": "#3A3340"}},
    ))
    figure.update_layout(
        height=300, margin={"l": 8, "r": 8, "t": 8, "b": 8},
        showlegend=False, yaxis={"title": i18n.t("¢ per share", "美分/股"), "zeroline": True},
    )
    st.plotly_chart(figure, use_container_width=True)

    cumulative = attribution.by_day[["day", *metrics.ATTRIBUTION_TERMS]].copy()
    for name in metrics.ATTRIBUTION_TERMS:
        cumulative[name] = cumulative[name].cumsum() * 100
    st.caption(i18n.t("Running total by day, in cents — when each greek did its work",
                      "按日累计(美分)— 每个 Greek 何时出的力"))
    st.line_chart(
        with_viewer_index(cumulative, "day"), height=200,
        color=[BRAND_GREEN, "#F5B342", BRAND_PURPLE, "#5AA9E6"],
    )

    # WHAT A DAY MEANS HERE, said in the interface and not only in the code.
    # This product can be collected by hand, and then "a day" is whatever the
    # person happened to press: today at 17:40, tomorrow at 11:05. Theta is
    # charged per calendar day and stays honest; the delta term does not, and
    # the difference lands in the residual looking like noise in the data rather
    # than a consequence of the schedule.
    spans = attribution.by_day["missing_trading_days"]
    st.caption(
        i18n.t(
            f"A day is the LAST collection of each trading day, New York time — "
            f"{len(attribution.by_day)} interval(s) in this window. Collect on a schedule for "
            "these numbers to mean what they say.",
            f"这里的“一天”指每个交易日(纽约时间)的最后一次采集 — 本窗口共 "
            f"{len(attribution.by_day)} 个区间。保持定时采集,这些数字才有意义。",
        )
    )
    notes = []
    if attribution.dropped_cheap:
        notes.append(
            i18n.t(
                f"{attribution.dropped_cheap} day(s) left out: the contract traded below "
                f"{config.ATTRIBUTION_MIN_PRICE * 100:.0f}c there, where implied volatility "
                "inverted from a one-cent quote makes every greek meaningless.",
                f"剔除了 {attribution.dropped_cheap} 天:这些日子合约价低于 "
                f"{config.ATTRIBUTION_MIN_PRICE * 100:.0f} 美分,一美分报价反推出的 IV "
                "会让每个 Greek 都失去意义。",
            )
        )
    if attribution.dropped_unreliable:
        notes.append(
            i18n.t(
                f"{attribution.dropped_unreliable} day(s) left out: their implied volatility "
                "failed the outlier guard, so the greeks are missing rather than wrong.",
                f"剔除了 {attribution.dropped_unreliable} 天:其隐含波动率未通过离群值校验,"
                "因此这些日子的 Greek 是缺失而非错误。",
            )
        )
    if int(spans.sum()):
        notes.append(
            i18n.t(
                f"{int(spans.sum())} trading day(s) inside the window were never collected — "
                "their moves are folded into the neighbouring interval, not counted as zero. "
                "The gamma term approximates a move locally; over a stretched interval it "
                "approximates less well, and the difference lands in the residual.",
                f"窗口内有 {int(spans.sum())} 个交易日从未被采集 — 其涨跌被并入相邻区间,"
                "而不是记为零。Gamma 项是对局部波动的近似,区间拉长后近似变差,"
                "差值体现在残差里。",
            )
        )
    residual_share = abs(totals["residual"]) / max(abs(totals["actual"]), 1e-9)
    if residual_share > 0.5:
        notes.append(
            i18n.t(
                f"The residual is {residual_share:.0%} of the whole move: the model does not "
                "explain this contract — the spread, a stale print or a rate move did more "
                "than the greeks.",
                f"残差占整个变动的 {residual_share:.0%}:模型解释不了这个合约 — "
                "点差、过期成交或利率变动的影响超过了 Greeks。",
            )
        )
    for note in notes:
        st.caption(f"⚠️ {note}")


def waterfall_labels(start: float, steps: list[float]) -> list[str]:
    """Bar labels for the waterfall, including the total.

    Computed here rather than from the `y` array, because Plotly IGNORES the y
    value of a `measure="total"` bar and derives its height itself — so the
    array carries a zero there, and a label built from it announces a contract
    worth nothing. The total is the sum, and this is the one place that says so.
    """
    return [f"{start:.1f}¢", *(f"{value:+.1f}¢" for value in steps),
            f"{start + sum(steps):.1f}¢"]


def format_compact(value: float) -> str:
    """1234 -> "1.2k", 1_500_000 -> "1.5m" — shorter than "1,234,567", so more
    expiry columns fit on screen without horizontal scrolling."""
    if pd.isna(value) or value == 0:
        return "0"
    sign = "-" if value < 0 else ""
    magnitude = abs(value)
    if magnitude >= 1_000_000:
        return f"{sign}{magnitude / 1_000_000:.1f}m"
    if magnitude >= 1_000:
        return f"{sign}{magnitude / 1_000:.1f}k"
    return f"{sign}{magnitude:.0f}"


GEX_CMAP = LinearSegmentedColormap.from_list("gex_dark", ["#b833e0", "#0d0d0d", "#22c55e"])

# Brand accent pair, reused across every chart below instead of Streamlit's
# default blue palette — purple as the primary series, green as the
# secondary/comparison one, so the color language stays consistent with the
# heatmap and the gammagrid.io teaser. The IV surface deliberately stays on
# Viridis rather than a brand gradient: a perceptually-uniform scale reads
# far better on a continuous 3-D surface.
BRAND_PURPLE = "#B833E0"
BRAND_GREEN = "#22C55E"


def style_gex_matrix(matrix: pd.DataFrame, atm_strike: float | None = None):
    """Zero-centered color scale on a BLACK background (custom diverging palette
    `GEX_CMAP`, not the stock PRGn with its light center) — matches the app's
    dark theme and the reference tool. Text (abbreviated, `format_compact`) and
    color are computed separately (`gmap`) — so the text can be shortened (1.2k
    instead of 1,234) independently of the exact numbers, which stay in the
    source matrix solely for color computation.

    `vmax` is taken not as the true maximum but as the 90th percentile of
    nonzero |values| — otherwise a single strong outlier (common in GEX: one
    ATM cell an order of magnitude above its neighbors) stretches the scale so
    far that nearly all other cells become indistinguishably washed out. Cells
    beyond vmax are simply painted the limit color — expected behavior for a
    heatmap; the exact number is still visible in the cell text.

    If `atm_strike` is given, the row closest to the underlying price is marked
    with a "➤" marker in the index and highlighted as a whole. Streamlit does
    not marshal index styles from a pandas Styler (`cellstyle_index` from
    `Styler._translate()` is simply dropped — verified against the
    `pandas_styler_utils.py` sources), so `Styler.map_index()` won't work for
    this: it has to be index text (guaranteed to display) plus styling of
    regular data cells (`cellstyle` is marshaled, that works)."""
    display = matrix.rename(columns=format_date)
    numeric = display.to_numpy(dtype=float)
    nonzero_abs = np.abs(numeric[numeric != 0])
    vmax = np.percentile(nonzero_abs, 90) if nonzero_abs.size else 0
    vmax = vmax if vmax > 0 else 1

    text = display.map(format_compact)

    # Convert the ENTIRE index to strings (not just the ATM row) — mixing floats
    # and a single str label in one Index yields an object-dtype column that
    # pyarrow cannot serialize directly (ArrowInvalid when rendering in
    # Streamlit, visible only in container logs — the widget itself suppresses
    # the error with an auto-fix, but that's extra fragility and an extra
    # traceback in the logs on every render).
    atm_label = None
    if atm_strike is not None and atm_strike in text.index:
        atm_label = f"➤ {atm_strike:g}"
    text.index = [atm_label if v == atm_strike else f"{v:g}" for v in text.index]

    styler = text.style.background_gradient(
        cmap=GEX_CMAP, vmin=-vmax, vmax=vmax, gmap=numeric, axis=None, text_color_threshold=0.6
    )
    if atm_label is not None:
        def highlight_atm_row(row: pd.Series) -> list[str]:
            style = "border-top: 2px solid #f59e0b; border-bottom: 2px solid #f59e0b; font-weight: 700;"
            return [style if row.name == atm_label else "" for _ in row]

        styler = styler.apply(highlight_atm_row, axis=1)
    return styler


def render_tradingview_widget(ticker: str, height: int = 300) -> str:
    """Auxiliary visualization (not a data source for the app — the focus stays
    on options analytics). The ticker is passed as-is, with no exchange-prefix
    mapping (NASDAQ:/AMEX:/...) — TradingView resolves the symbol itself; if
    some ticker resolves to the wrong thing, we'll deal with it then."""
    config_json = json.dumps({
        "width": "100%",
        "height": height,
        "symbol": ticker,
        "interval": "D",
        "timezone": "Etc/UTC",
        "theme": "dark",
        "style": "1",
        "locale": "en",
        "hide_legend": True,
        "hide_top_toolbar": False,
        "hide_side_toolbar": True,
        "allow_symbol_change": False,
        "save_image": False,
        "support_host": "https://www.tradingview.com",
    })
    # autosize:true did not stretch the canvas to the container's full height in
    # this (nested iframe) context — it left an empty tail at the bottom.
    # Explicit width/height in the widget config instead of autosize fixes
    # exactly that.
    return f"""
    <div class="tradingview-widget-container" style="height:{height}px;width:100%;margin:0;padding:0">
      <div class="tradingview-widget-container__widget" style="height:100%;width:100%"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
      {config_json}
      </script>
    </div>
    """


conn = db.get_connection()

# The teaser's grid background and mono display type, applied globally so
# the app shares the gammagrid.io visual language instead of reading as bare
# default Streamlit. Deliberately NOT touching st.dataframe/st.data_editor
# styling: dense tables need to stay plain and readable, not branded —
# over-styling them would visually overload the page.
st.markdown(
    """
    <style>
      .stApp {
        background-image:
          linear-gradient(#1C2320 1px, transparent 1px),
          linear-gradient(90deg, #1C2320 1px, transparent 1px);
        background-size: 64px 64px;
        background-position: center -1px;
      }
      [data-testid="stHeading"] h1,
      [data-testid="stHeading"] h2,
      [data-testid="stHeading"] h3 {
        font-family: "SF Mono","JetBrains Mono","IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace;
      }
      [data-testid="stTab"] p {
        font-family: "SF Mono","JetBrains Mono","IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace;
        letter-spacing: .02em;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header wordmark: the same 2x2 diagonal chip mark as the favicon, next to the
# title, matching the teaser site and the hosted product's header — small,
# because the header is a place to know where you are rather than a page
# banner, and the screen belongs to the chain below it.
st.markdown(
    f"""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:0.25rem;">
      <div style="display:grid;grid-template-columns:13px 13px;grid-template-rows:13px 13px;gap:3px;flex-shrink:0;">
        <div style="background:{theme.PRIMARY};"></div>
        <div style="background:{theme.BORDER};"></div>
        <div style="background:{theme.BORDER};"></div>
        <div style="background:{theme.ACCENT};"></div>
      </div>
      <span style="font-size:{theme.TEXT_STRONG};font-weight:700;letter-spacing:0.01em;">GammaGrid</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(
    i18n.t(
        "Open-source options positioning dashboard — dealer gamma exposure, max pain, "
        "open interest, and IV surface for your whole watchlist.",
        "开源期权持仓结构仪表盘 — 做市商 Gamma 敞口、最大痛点、持仓量与波动率曲面,覆盖你的整个自选列表。",
    )
)

with st.sidebar:
    i18n.toggle()
    st.header(i18n.t("Watchlist", "自选列表"))
    watchlist = db.get_watchlist(conn)

    # ONE BOX THAT FILTERS AS YOU TYPE, and it replaced a plain text field for a
    # reason that has nothing to do with looks. A field you can only type into
    # is a field you can only be wrong in: it cannot tell somebody that what
    # they want exists under another name, because it does not know what
    # exists. `APPL`, `NVIDIA`, `NASDAQ` and `BTCUSD` all have the shape of a
    # ticker, and all of them used to be accepted and then fail forever.
    #
    # A SELECTBOX HOLDING THE WHOLE CATALOGUE FILTERS IN THE BROWSER, on every
    # keystroke, and empties the moment the box is cleared. A server-side search
    # cannot: Streamlit hands a text field's value to Python only when it loses
    # focus, so somebody typing "Apple" would see nothing at all until they
    # clicked elsewhere — and if they went straight for the Add button instead,
    # they would never see a suggestion.
    #
    # `accept_new_options` makes the same widget the fallback for a symbol the
    # directory does not list, so there is one door instead of two — and it is
    # what keeps this working on an installation whose catalogue is empty
    # because the download failed or the worker has never run.
    #
    # `filter_mode="contains"` rather than the default fuzzy match: fuzzy
    # re-sorts by its own score and throws away the ordering this list was built
    # with, and that ordering is most of what makes the search useful.
    _options = _cached_catalogue_options(conn, tuple(watchlist))
    _label_to_symbol = {label: symbol for symbol, label in _options}
    # The directory as a set — the only statement this product holds that a
    # string is a real security with listed options. It is what tells "AAPX,
    # deliberately typed" apart from "APPL, mistyped", and what resolves SAP.DE
    # to SAP without a map of European companies.
    _catalogue_symbols = set(_label_to_symbol.values())
    _choice = st.selectbox(
        i18n.t("Add ticker", "添加代码"),
        options=list(_label_to_symbol),
        index=None,
        placeholder=i18n.t("AAPL or Apple", "AAPL 或 Apple"),
        accept_new_options=True,
        filter_mode="contains",
        key="add_ticker_choice",
        help=(
            i18n.t(
                "Search by symbol or company name, or type a symbol that is not "
                "listed. Data comes from Yahoo Finance (yfinance), and not every "
                "listed chain is available there. US-listed options only — European "
                "and Asian listings (SAP.DE, ASML.AS) have no options data here, but "
                "most large non-US companies also trade in the US as ADRs, which do: "
                "try ASML, SAP, SHEL, NVO, BUD.",
                "按代码或公司名搜索,也可直接输入目录里没有的代码。数据来自 Yahoo Finance"
                "(yfinance),并非所有挂牌期权链都在其覆盖范围内。仅支持美股挂牌期权 — "
                "欧亚交易所上市品种(SAP.DE、ASML.AS)在此无期权数据,但多数大型非美公司"
                "同时以 ADR 形式在美交易,这些有数据:试试 ASML、SAP、SHEL、NVO、BUD。",
            )
        ),
    )
    new_ticker = (_label_to_symbol.get(_choice) or (_choice or "")).strip().upper()

    # Said once, and only while it is true. An empty directory is not a failure
    # anybody has to act on — the box still accepts anything typed into it — but
    # somebody who expected to search for "Apple" and got nothing deserves to
    # know it is the catalogue that has not arrived, not their spelling.
    if not _catalogue_symbols:
        st.caption(
            i18n.t(
                "The symbol directory has not been downloaded yet, so this box cannot "
                "search by company name — type the symbol itself. It arrives on the "
                "collector's next pass.",
                "代码目录尚未下载完成,暂时无法按公司名搜索 — 请直接输入代码。采集器下一轮"
                "运行时会自动补上。",
            )
        )

    depth = _cached_collection_depth()

    # ONE ANSWER TO ONE QUESTION. Three mechanisms used to speak here — the
    # provider's refusal list, the map of instruments with no US options, and
    # the typo search — in three places, in three voices, and with holes between
    # them: SAP.DE, BTCUSDT and 9988.HK got nothing at all while the help text
    # under this very box named ADRs as the answer.
    #
    # ASKED ON EVERY KEYSTROKE AND NEVER OVER THE NETWORK. Everything it needs
    # is already in memory — the provider's declared refusals, the catalogue
    # just handed to the widget, and the tickers already collected — so somebody
    # who typed SAP.DE reads the answer instead of waiting seconds for Yahoo to
    # say the same no.
    _provider = providers.get_provider()
    _unsupported_here = getattr(_provider, "unsupported_symbols", {}) or {}
    _proposal = suggestions.propose(
        new_ticker,
        depth.keys(),
        in_catalogue=_catalogue_symbols.__contains__,
        unsupported=_unsupported_here,
        provider=_provider.name,
    )
    # ONLY THE SOURCE'S OWN REFUSAL TAKES THE BUTTON AWAY. The rest is this
    # product's reading of what somebody meant, and being wrong about that must
    # cost them a sentence, not the ability to try.
    if _proposal:
        st.warning(_proposal.message)

    # WHAT THIS TELLS SOMEBODY BEFORE THEY COMMIT, and it is said only when
    # there is something to commit to. Two different things, and both used to be
    # silence:
    #
    #   * a symbol with no chain to collect (APPL, NASDAQ, BTCUSD) was accepted
    #     without a word and then failed on every cycle forever;
    #   * a symbol that IS valid still starts its history today, so the first
    #     chart is a single point — on a fresh install, always. Somebody who was
    #     promised history and sees one dot concludes the tool is broken, and
    #     they are not wrong to.
    #
    # AFTER THE PROPOSAL, AND NOT FOR A SYMBOL THE SOURCE HAS REFUSED. Promising
    # that history "starts at the first collection" directly above a sentence
    # saying there will never be one is the product contradicting itself in two
    # consecutive lines.
    if new_ticker and not (_proposal and _proposal.blocking):
        since = depth.get(new_ticker)
        if since:
            st.caption(i18n.t(f"Already collected since {since:%d %b %Y}.",
                              f"自 {since:%Y-%m-%d} 起已有采集。"))
        else:
            st.caption(
                i18n.t(
                    "History for a new ticker starts at the first collection — the first "
                    "chart will have one point, and fills in from there.",
                    "新代码的历史从第一次采集开始 — 第一张图只有一个点,之后随采集逐渐补全。",
                )
            )

    if st.button(i18n.t("Add", "添加"), disabled=not new_ticker or bool(_proposal and _proposal.blocking)):
        # THE LIVE CHECK STILL RUNS, and the catalogue does not replace it: the
        # directory says a symbol has listed options somewhere, not that Yahoo
        # will serve its chain. `is False` and not `not ...` — None means the
        # source could not answer, and a ticker must never be refused on that. A
        # false negative here looks like the product being broken; a false
        # positive shows up in the collection log within the hour and is
        # suspended by itself after six.
        if _cached_has_options(new_ticker) is False:
            # A symbol the directory lists is not "not a ticker". The limitation
            # is ours, and saying otherwise costs the message its credibility
            # with exactly the person who knows the symbol is real.
            st.error(
                suggestions.source_refusal(
                    new_ticker, _unsupported_here.get(new_ticker), _provider.name
                )
                if new_ticker in _catalogue_symbols
                else suggestions.refusal(
                    new_ticker, watchlist, _catalogue_symbols.__contains__
                )
            )
        else:
            db.add_ticker(conn, new_ticker)
            _cached_collection_depth.clear()
            _cached_catalogue_options.clear()
            st.rerun()

    # Symbols that stopped being asked for. Marked in the list itself rather
    # than only in the log: the watchlist is where somebody looks to find out
    # what the tool is doing, and "quietly not collecting" is indistinguishable
    # from "collecting fine" without a mark here.
    suspended = db.unresolvable_tickers(conn)
    current_ticker = st.session_state.get("selected_ticker")
    for ticker in watchlist:
        col1, col2 = st.columns([3, 1])
        if col1.button(
            f"⏸ {ticker}" if ticker in suspended else ticker,
            key=f"select_ticker_{ticker}",
            type="primary" if ticker == current_ticker else "secondary",
            use_container_width=True,
            help=(
                i18n.t(
                    f"Not being collected: every attempt has failed since "
                    f"{suspended[ticker]:%d %b %Y}, and nothing has ever been collected for it. "
                    "Check the spelling, or remove it. Nothing has been deleted.",
                    f"未在采集:自 {suspended[ticker]:%Y-%m-%d} 起每次尝试均失败,且从未采集到过数据。"
                    "请检查拼写,或将其移除。不会删除任何已有数据。",
                )
            ) if ticker in suspended else None,
        ):
            st.session_state["selected_ticker"] = ticker
            st.rerun()
        if col2.button("✕", key=f"remove_{ticker}"):
            db.remove_ticker(conn, ticker)
            if st.session_state.get("selected_ticker") == ticker:
                # otherwise on the next run the selectbox would receive a value
                # that is no longer among its options, and crash
                st.session_state.pop("selected_ticker", None)
            st.rerun()

    st.divider()
    if st.button(
        i18n.t("Collect data", "立即采集"),
        type="primary",
        disabled=not watchlist,
        help=i18n.t(
            "Best run while the US options market is open. Outside market "
            "hours, Yahoo Finance often reports open_interest=0 across the whole "
            "chain, which fails the collection for that ticker.",
            "最好在美股期权开市时段运行。休市时段 Yahoo Finance 常把整条链的 "
            "open_interest 报成 0,导致该次采集被判为无效。",
        ),
    ):
        with st.spinner(i18n.t("Collecting data...", "正在采集数据...")):
            results = collector.collect_watchlist(conn, watchlist)
        for ticker, status in results.items():
            if status == "success":
                st.success(f"{ticker}: {i18n.t('OK', '成功')}")
            elif status.startswith("skipped"):
                # A deliberate decision not to try is not an error, and dressing
                # it in red teaches people to ignore the red.
                st.warning(f"{ticker}: " + i18n.t("skipped", "跳过")
                           + status.removeprefix("skipped"))
            else:
                st.error(f"{ticker}: {status}")

    # Scheduled collection. The list is short on purpose: the floor is set by
    # the data source rather than by preference, and a free-text box inviting
    # "every minute" invites getting blocked. Off is the default — a tool that
    # starts hitting an API the moment it is installed made a decision that was
    # not its to make.
    current_interval = db.get_collector_interval(conn)
    labels = list(config.COLLECTOR_INTERVAL_CHOICES)
    current_label = next(
        (name for name, minutes in config.COLLECTOR_INTERVAL_CHOICES.items()
         if minutes == current_interval),
        labels[0],
    )
    chosen = st.selectbox(
        i18n.t("Collect automatically", "自动采集"),
        labels,
        index=labels.index(current_label),
        help=i18n.t(
            "Runs in the background worker container, not in this page — it keeps "
            "collecting while nobody is looking, and stops when you stop the containers.",
            "采集由后台 worker 容器执行,与本页面无关 — 无人查看时也会持续采集,"
            "容器停止时随之停止。",
        ),
    )
    chosen_minutes = config.COLLECTOR_INTERVAL_CHOICES[chosen]
    if chosen_minutes != current_interval:
        db.set_collector_interval(conn, chosen_minutes)
        st.rerun()

    # The cost, stated at the moment of the decision and computed from THIS
    # watchlist rather than from a generic figure — chain sizes differ by an
    # order of magnitude between one small ticker and a few index ETFs.
    if chosen_minutes > 0:
        growth = db.estimated_growth_mb_per_month(conn, chosen_minutes)
        if growth > 0:
            st.caption(
                i18n.t(
                    f"About {growth / 1000:.1f} GB per month at this interval, for your current "
                    f"watchlist. Nothing is ever deleted; expired contracts move to a separate "
                    f"table after {config.CONTRACT_ARCHIVE_GRACE_DAYS} days.",
                    f"按当前频率与自选列表估算,约每月 {growth / 1000:.1f} GB。"
                    f"数据永不删除;到期合约在 {config.CONTRACT_ARCHIVE_GRACE_DAYS} 天后归档至单独表。",
                )
            )
        else:
            st.caption(
                i18n.t(
                    "Collect once first and this will say how much disk a month of collection "
                    "costs for your watchlist.",
                    "先手动采集一次,这里就会给出你的自选列表每月采集所占的磁盘估算。",
                )
            )
        # Said here rather than in a FAQ, because this is where somebody picks a
        # frequency and works out what it will cost.
        #
        # This comment used to claim the estimate above "already assumes the
        # market is shut most of the week". It did not — the formula multiplied
        # by 24 hours and 30 days, and overstated the answer roughly fivefold.
        # The sentence was written in the same change that added the calendar:
        # the consequence for the neighbouring function was reasoned out in
        # words and never carried into the code. A comment asserting a property
        # the code does not have is worse than no comment, because it is what
        # the next reader checks INSTEAD of the formula.
        st.caption(
            i18n.t(
                "**Outside market hours the collector does not run.** US options trade "
                "09:30–16:00 New York on weekdays; while the market is shut the chain does not "
                "change, so one snapshot per closed day is taken as a safety net and the rest "
                "are skipped — roughly four fifths of the week not spent on duplicates.",
                "**休市时段采集器不运行。**美股期权在交易日 09:30–16:00(纽约时间)交易;"
                "休市期间到期链不变,因此每个休市日仅保留一份兜底快照,其余全部跳过 — "
                "一周约五分之四的时间不浪费在重复快照上。",
            )
        )
    else:
        st.caption(i18n.t("Automatic collection is off. Use the button above to collect on demand.",
                          "自动采集已关闭。需要时用上方按钮手动采集。"))

    # A cooldown that nobody can see is indistinguishable from the tool being
    # broken — and worse, from the market simply being closed, which is a state
    # this sidebar already describes a few lines above. Said in full: what
    # happened, and when it stops.
    cooldown_until = db.provider_cooldown_until(conn)
    if cooldown_until is not None:
        st.warning(
            i18n.t(
                f"**The data source is limiting requests.** Collection is paused until "
                f"{to_viewer(pd.Series([cooldown_until])).dt.strftime('%H:%M').iloc[0]} "
                f"({timezone_label()}). This is not the market being closed — it is Yahoo "
                "Finance refusing requests from this installation, usually after too many "
                "in a short time. Nothing is lost; the next pass collects as usual.",
                f"**数据源正在限流。**采集暂停至 "
                f"{to_viewer(pd.Series([cooldown_until])).dt.strftime('%H:%M').iloc[0]} "
                f"({timezone_label()})。这不是休市 — 是 Yahoo Finance 短时间内请求过多而"
                "拒绝本安装的请求。不丢任何数据,下一轮照常采集。",
            )
        )

    with st.expander(i18n.t("📋 Collection log", "📋 采集日志")):
        st.caption(i18n.t(f"Times in your own timezone ({timezone_label()}).",
                          f"时间为你所在时区({timezone_label()})。"))
        runs = db.get_recent_runs(conn, limit=30)
        if runs.empty:
            st.caption(i18n.t("No collections yet.", "还没有采集记录。"))
        else:
            runs_display = runs.copy()
            runs_display["started_at"] = to_viewer(runs_display["started_at"]).dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            runs_display["oi_zero_fraction"] = runs_display["oi_zero_fraction"].map(
                lambda v: f"{v:.0%}" if pd.notna(v) else "—"
            )
            st.dataframe(
                runs_display[
                    ["started_at", "ticker", "status", "rows_fetched", "oi_zero_fraction", "error_message"]
                ],
                use_container_width=True,
                hide_index=True,
            )
        st.caption(
            i18n.t(
                "`oi_zero_fraction` — the fraction of contracts with open_interest=0 in the "
                f"collected chain. Above {config.MAX_ZERO_OI_FRACTION:.0%} the snapshot is "
                "considered suspect and is not saved (status=failed, reason in `error_message`), "
                "even if the request to the data source itself completed without a network error. "
                "The most common cause is collecting outside regular market hours — Yahoo Finance "
                "often reports open_interest=0 across the whole chain while the market is closed. "
                "If a ticker keeps failing, try again during market hours.",
                "`oi_zero_fraction` — 本次采集的链中 open_interest=0 的合约占比。"
                f"超过 {config.MAX_ZERO_OI_FRACTION:.0%} 即视为可疑快照,不予保存"
                "(status=failed,原因见 `error_message`),即使数据源请求本身没有网络错误。"
                "最常见原因是休市时段采集 — 休市时 Yahoo Finance 常把整条链的持仓量报为 0。"
                "若某代码持续失败,请开市时段再试。",
            )
        )

    st.divider()

    # A button, deliberately, and deliberately not a dialog. Measured over the
    # fortnight to 18.08.2026: 64 people cloned this repository and 11 starred
    # it. The ask does exist in the README — below the licence section, where
    # nobody reading a quick start ever arrives. Somebody with the app open in
    # front of them has already got further than any reader of the page.
    #
    # It sits here permanently rather than appearing after a successful
    # collection: a prompt that waits for the moment you are pleased and then
    # interrupts it is a trick, and the same trick costs us the collection
    # result on the screen underneath. A button that is always in the same place
    # can be pressed when the tool has earned it and ignored forever otherwise.
    st.link_button(
        "⭐ Star GammaGrid on GitHub",
        "https://github.com/gammagrid/gammagrid",
        use_container_width=True,
        help="Stars are how people find this project — there is no marketing budget behind it.",
    )
    st.caption("Questions or feedback: [hello@gammagrid.io](mailto:hello@gammagrid.io)")
    st.caption(i18n.t(
        "Based on open-source GammaGrid (AGPL-3.0), with a bilingual layer added.",
        "基于开源项目 GammaGrid(AGPL-3.0)二次开发,新增中英双语界面。"))
    # The revision, in the footer of the sidebar rather than in the header.
    # Every self-hosted installation is a different version — people update
    # when they feel like it and do not remember when — so a screenshot with no
    # version in it starts every support conversation with a round of
    # correspondence. One arrived with two collection errors and had to be
    # dated by which caption was MISSING from it.
    #
    # Here specifically because this is the corner people photograph: the
    # collection log is directly above, and it is what they are looking at when
    # something has gone wrong. The header gets cropped.
    st.caption(f"GammaGrid {config.APP_VERSION}")

if not watchlist:
    st.info(i18n.t("Add a ticker to the watchlist on the left to get started.",
                   "在左侧自选列表添加一个代码即可开始。"))
    st.stop()

selected_ticker = st.selectbox(i18n.t("Ticker to analyze", "分析标的"), watchlist, key="selected_ticker")

# Bounded by default, never a permanent cap on what you can see: collected data
# is the most valuable thing in the app, and a rolling window must not silently
# hide a long-lived contract's early history — an 18-month-old two-year LEAP
# would lose its first year. days=None below fetches everything.
load_full_history = st.checkbox(
    i18n.t(f"Load full history (default: last {config.SNAPSHOT_HISTORY_DAYS} days)",
           f"加载全部历史(默认:最近 {config.SNAPSHOT_HISTORY_DAYS} 天)"),
    key="load_full_history",
    help=i18n.t("Slower on tickers with a long collection history.",
                "采集历史越长的代码,加载越慢。"),
)
history_days = None if load_full_history else config.SNAPSHOT_HISTORY_DAYS

# NO VIEW READS THE WHOLE TICKER. What used to be here was one frame of every
# row this ticker has, loaded on every interaction and filtered in pandas by
# whichever view was showing — which is fine while collection is a button and
# stops being fine the moment it runs on a schedule. Measured on the hosted
# product, which reached that point first: 3.1M rows, 22 seconds and 0.8 GB of
# memory for one liquid ticker, to draw charts a few hundred points wide.
#
# Each view now asks for what it shows: one snapshot, two calendar days, one
# contract, or a stored rollup. `db.get_snapshots` remains for tooling; if it
# reappears in a view, this is what regressed.
#
# ONE SOURCE PER PAGE, resolved once here and passed to every read below rather
# than looked up inside each of them: a render then pays for one small query
# instead of a dozen identical ones. GammaGrid never blends two providers on
# one screen — see db.active_source for why implied volatility makes that a
# correctness problem and not a tidiness one.
page_source = db.active_source(conn, selected_ticker)


def with_our_iv(frame):
    """Every chain this page draws goes through here.

    THE VOLATILITY IS SOLVED FROM THE PRICE, not taken from the data source.
    Yahoo — the source this product ships with, and for most installations the
    only one — leaves implied volatility missing on part of a chain and reports
    numbers that do not reproduce the quoted price on another part. Every greek,
    the GEX profile, the screener's filters and the surface are built on that
    column, so a chain with holes in it is a screen with holes in it.

    Applied after the read rather than inside it, and applied to a frame that
    was read for one moment or one contract: solving costs single-digit
    milliseconds on a chain of a few thousand contracts, which is cheap enough
    to redo on every rerun and far cheaper than a cache keyed on something that
    can go stale.

    WHERE THE PRICE DOES NOT DETERMINE A VOLATILITY, THE SOURCE'S NUMBER
    STANDS. Refusing there would empty part of a liquid chain off the screener
    and the profile, and a hole reads as a broken product rather than as an
    honest silence. Which rows are ours is recorded in `iv_is_ours`, and the
    source's own number is kept in `provider_implied_volatility` — throwing it
    away would make "how far apart are the two models" unanswerable.
    """
    return metrics.with_solved_iv(frame, metrics.DEFAULT_PRICING, ticker=selected_ticker)


latest_df = with_our_iv(db.get_latest_snapshot(conn, selected_ticker, source=page_source))

# HOW OLD THIS SCREEN IS, SAID BEFORE ANY OF IT IS DRAWN. Two questions, asked
# in this order because the answers rule each other out: is this ticker's
# collection healthy, and — only if there is nothing to report about it — what
# is the market doing.
#
# The freshness note speaks about THIS symbol and wins, because "collection for
# GLD is failing" is a fact about the screen being read, while "market closed"
# is a fact about the day. Saying both would print two lines that agree in tone
# and disagree in meaning, and a reader picks the calmer one.
#
# The app never asks the provider itself: a page render has no business making a
# network call, and two processes asking separately would eventually disagree in
# front of the reader. Both of these read what the collector already wrote.
_collector_interval = max(db.get_collector_interval(conn), 15)
_last_success, _failures = db.collection_state(conn, selected_ticker, source=page_source)
_freshness = freshness.assess(_last_success, _failures, _collector_interval)

if _freshness:
    _state, _note = _freshness
    if _state == freshness.RESTING:
        st.caption(_note)
    else:
        # STALE and FAILING are the two the eye must not slide past, so they
        # keep the banner. RESTING is the market being shut, and a warning
        # there is the product crying wolf every weekend.
        st.warning(_note, icon="⚠️")
else:
    # What the collector last believed about the market, phrased by the
    # calendar. Only reached when this ticker has nothing wrong with it.
    _market_note = market_calendar.status_note(
        db.get_setting(conn, "market_state"),
        db.get_setting(conn, "market_state_at"),
        _collector_interval,
    )
    if _market_note:
        st.caption(_market_note)

# A SYMBOL THIS SOURCE WILL NOT SERVE, said before anything else on the page.
# Without it the screen is indistinguishable from a ticker that has simply not
# been collected yet — and the person looking at it typed a symbol they know is
# real, so "no data, click Collect" reads as the product being broken. The
# substitute comes from the provider, so it is right by construction rather
# than by a list somebody has to remember to update.
_refused_here = getattr(providers.get_provider(), "unsupported_symbols", {}) or {}
if selected_ticker in _refused_here:
    st.error(
        suggestions.source_refusal(
            selected_ticker, _refused_here[selected_ticker], providers.get_provider().name
        ),
        icon="⛔",
    )

if latest_df.empty:
    # NOT "click Collect data" WHEN COLLECTING IS THE THING THAT IS FAILING.
    # The note above already said the source has refused this symbol several
    # times; inviting somebody to press the button again directly underneath it
    # is the product contradicting itself in two consecutive lines, and the
    # press costs another refused request.
    if not (_freshness and _freshness[0] == freshness.FAILING):
        st.info(i18n.t(f"No data for {selected_ticker}. Click “Collect data” on the left.",
                       f"{selected_ticker} 暂无数据。请点击左侧“立即采集”。"))
    st.stop()

# Said out loud only when there is something to say. A person who switched
# providers sees their older history disappear from the charts, and hidden
# history with no explanation is indistinguishable from data loss.
_ticker_sources = db.sources_for(conn, selected_ticker) if page_source else []
if len(_ticker_sources) > 1:
    others = ", ".join(s for s in _ticker_sources if s != page_source)
    st.caption(
        i18n.t(
            f"Showing data from **{page_source}**. This ticker also has history from {others}, "
            "which is kept but not shown: two providers calculate implied volatility "
            "differently, so mixing them on one chart would draw moves that never happened.",
            f"当前显示 **{page_source}** 的数据。该代码还有来自 {others} 的历史,"
            "已保留但未显示:两家数据源的隐含波动率算法不同,混在一张图上会画出从未发生过的波动。",
        )
    )

latest_date = latest_df["collected_at"].max()
expiries = sorted(latest_df["expiry"].unique())

# Deliberately not st.tabs. Streamlit has no lazy tabs: the body of every
# `with tab_x:` runs on every rerun and the browser merely hides the seven you
# are not looking at, so each interaction pays for eight views to show one.
# Measured on a realistically-shaped chain (3,520 contracts per snapshot), one
# render is 1.3-1.7s — and tripling the stored rows barely moved that, which
# says the cost is the tabs, not the data, and will not improve on its own.
#
# The trade this makes, stated plainly: switching views used to be instant
# (everything was already in the browser) and now costs one script run. That
# run renders a single view, so it is a fraction of what every interaction used
# to cost.
# Pinned contracts are needed by two views — the Contract tab and the contract
# card the Screener opens on a row click. This used to be fetched inside the
# Contract tab's body and read from the Screener's, which worked only because
# st.tabs executed every tab on every run. With one view rendering at a time
# that is a NameError on the Screener, reachable only after clicking a row —
# the sibling project shipped exactly that and a user found it in the browser.
_tracked_cache: dict[str, pd.DataFrame] = {}


def get_tracked() -> pd.DataFrame:
    if "df" not in _tracked_cache:
        _tracked_cache["df"] = db.get_tracked_contracts(conn, selected_ticker)
    return _tracked_cache["df"]


# WHAT THE WHOLE CHAIN ADDS UP TO, IN ONE LINE, ABOVE EVERY VIEW. Eight views
# answer eight questions and none of them answers the first one somebody has:
# is dealer hedging damping this or amplifying it today. The number that says
# so is net GEX, which is eight digits and a sign — and the sign is the part
# that matters, which no amount of staring at the digits teaches.
#
# BUILT ONCE, FROM THE SNAPSHOT ALREADY IN HAND. It folds one GEX matrix over
# the near-term expiries and reads three numbers off it, so drawing it on every
# view costs one matrix rather than one per view. The near term rather than the
# whole chain, on purpose: read over every expiry it would disagree with the
# heatmap directly below it on any symbol whose open interest sits far out.
_weather = metrics.gamma_weather(latest_df)
if _weather:
    st.markdown(weather.panel(_weather, latest_moment=latest_date), unsafe_allow_html=True)

VIEWS = [
    "Overview", "Changes", "Max Pain / GEX", "GEX Heatmap", "Volatility (IV)",
    "Contract", "Screener", "Unusual Activity", "OI Delta",
]
VIEW_ZH = {
    "Overview": "总览", "Changes": "每日变化", "Max Pain / GEX": "最大痛点 / GEX",
    "GEX Heatmap": "GEX 热力图", "Volatility (IV)": "波动率 (IV)", "Contract": "单合约",
    "Screener": "筛选器", "Unusual Activity": "异动", "OI Delta": "持仓变化",
}
_VIEW_EN = {i18n.t(v, VIEW_ZH[v]): v for v in VIEWS}
_view_labels = list(_VIEW_EN)
active_view = st.segmented_control(
    "View", _view_labels, default=_view_labels[0], key="active_view", label_visibility="collapsed"
)
# A segmented control returns None when the current pill is deselected; an
# empty page is never what that gesture means here. Comparisons below stay on
# the canonical English names — the label chosen above is mapped back first.
active_view = _VIEW_EN.get(active_view, VIEWS[0]) if active_view else VIEWS[0]

if active_view == "Overview":
    st.subheader(i18n.t("Overview", "总览"))
    st.caption(i18n.t(f"Data for {selected_ticker} collected {format_datetime(latest_date)}",
                      f"{selected_ticker} 数据采集于 {format_datetime(latest_date)}"))
    st.subheader(i18n.t("Put/Call Ratio", "Put/Call 比率"))
    # Aggregated in SQL rather than by grouping every raw row here: same
    # numbers, a fraction of the work. metrics.put_call_ratio remains the
    # definition of the ratio and the oracle the tests check this against.
    pcr = db.get_put_call_ratio(conn, selected_ticker, days=history_days, source=page_source)
    st.line_chart(with_viewer_index(pcr)[["pcr_volume", "pcr_oi"]], color=[BRAND_PURPLE, BRAND_GREEN])
    with st.expander(i18n.t("ℹ️ How to read this", "ℹ️ 怎么读这张图")):
        st.write(
            i18n.t(
                "The ratio of put to call volume/open interest. A value noticeably above "
                "0.7–1.0 usually signals a defensive/bearish stance, below — a bullish one. "
                "The absolute level matters less than a sharp deviation from the ticker's own "
                "history on this chart — compare the current value to past ones, not to "
                "general rules of thumb.",
                "Put 与 Call 的成交量/持仓量之比。明显高于 0.7–1.0 通常代表防守/看空,"
                "低于该区间则偏看多。绝对水平不如该代码自身历史上的急剧偏离重要 — "
                "和它自己的过去比,而不是套通用经验值。",
            )
        )

    st.subheader(i18n.t("Volatility surface (IV surface)", "波动率曲面 (IV Surface)"))
    iv_surface_points = metrics.iv_surface(latest_df)
    iv_grid = metrics.iv_surface_grid(iv_surface_points)
    if iv_grid is None:
        st.info(
            i18n.t(
                "Not enough data for a volatility surface — need at least 2 distinct strikes "
                "and 2 distinct expiries with known IV in the latest snapshot.",
                "数据不足以绘制波动率曲面 — 最新快照中至少需要 2 个不同行权价和 2 个不同到期、且 IV 已知。",
            )
        )
    else:
        grid_strikes, grid_years, grid_iv = iv_grid

        # The grid is built by linear interpolation — its x/y nodes (strike,
        # years) almost never coincide with actually traded contracts, and the
        # default tooltip would show meaningless fractional values. customdata
        # swaps them for practically useful ones: the nearest real strike from
        # the source points and DTE in days (years exist only for the
        # interpolation math — they're awkward to read).
        real_strikes = np.sort(iv_surface_points["strike"].unique())
        nearest_strike_per_col = np.array(
            [f"{real_strikes[np.abs(real_strikes - gx).argmin()]:g}" for gx in grid_strikes]
        )
        dte_per_row = np.round(grid_years * 365).astype(int).astype(str)
        customdata = np.dstack([
            np.tile(nearest_strike_per_col, (len(grid_years), 1)),
            np.tile(dte_per_row.reshape(-1, 1), (1, len(grid_strikes))),
        ])

        fig = go.Figure(
            data=[
                go.Surface(
                    x=grid_strikes,
                    y=grid_years,
                    z=grid_iv,
                    colorscale="Viridis",
                    customdata=customdata,
                    hovertemplate=(
                        "Strike ≈ %{customdata[0]}<br>"
                        "DTE: %{customdata[1]} d<br>"
                        "IV: %{z:.2f}<extra></extra>"
                    ),
                )
            ]
        )
        # The Y axis stays continuous in years-to-expiry — otherwise the
        # interpolation between expiries (uneven spacing, differing strike
        # counts) would make no sense. But its tick labels are the real expiry
        # dates rather than abstract fractions of a year: years exist for the
        # math, not for reading.
        expiry_ticks = (
            iv_surface_points[["years_to_expiry", "expiry"]]
            .drop_duplicates()
            .sort_values("years_to_expiry")
        )
        fig.update_layout(
            scene={
                "xaxis_title": i18n.t("Strike", "行权价"),
                "yaxis": {
                    "title": i18n.t("Expiry", "到期日"),
                    "tickmode": "array",
                    "tickvals": expiry_ticks["years_to_expiry"].tolist(),
                    "ticktext": [format_date(d) for d in expiry_ticks["expiry"]],
                },
                "zaxis_title": "IV",
            },
            margin={"l": 0, "r": 0, "t": 10, "b": 0},
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)
        with st.expander(i18n.t("ℹ️ How to read this", "ℹ️ 怎么读这张图")):
            st.write(
                i18n.t(
                    "The implied volatility surface across the whole chain at the latest "
                    "snapshot: strike on one axis, expiry date on the other, height/color — the "
                    "IV itself. For each strike the OTM contract's IV is used (put below the "
                    "current underlying price, call above): ITM contracts' IV is usually less "
                    "reliable due to low liquidity and wide spreads.\n\n"
                    "Real strikes and expiries rarely fall on a regular grid (different expiries "
                    "have different strike spacing and ranges, far dates are coarser), so the "
                    "surface is filled in by linear interpolation between the actual points (and "
                    "at the edges, where interpolation runs out of data, by nearest neighbor) to "
                    "avoid gaps/holes. This is display smoothing only, not new data — the source "
                    "points remain the app's own snapshots, with no calls to external sources. "
                    "Because of the interpolation, a grid node under the cursor almost never "
                    "matches an actually traded contract — the tooltip shows the nearest real "
                    "strike and DTE in days rather than the interpolated values.\n\n"
                    "The typical shape is a \"smile\" or a skew across strikes at each expiry "
                    "(see the Volatility (IV) tab for a single-expiry slice) and a rising/falling "
                    "overall IV level as expiry moves further out (term structure). Sharp local "
                    "spikes on the surface usually indicate sparse data in that region rather "
                    "than a real market effect — trust the overall shape more than individual "
                    "bumps.",
                    "最新快照下整条链的隐含波动率曲面:一条轴是行权价,另一条是到期日,"
                    "高度/颜色就是 IV 本身。每个行权价取虚值合约的 IV(标的价格下方取 Put、"
                    "上方取 Call):实值合约因流动性差、点差宽,IV 通常更不可靠。\n\n"
                    "真实行权价与到期日很少落在规则网格上(不同到期的档距与覆盖范围不同,"
                    "远月更稀疏),因此曲面由真实点之间的线性插值补齐(边缘插值无数据可用时"
                    "取最近邻),以避免空洞。这只是展示层面的平滑,不是新数据 — 数据点仍是"
                    "本应用自己的快照,不调用任何外部数据源。由于插值,光标下的网格点几乎"
                    "从不对应真实成交过的合约 — 悬浮提示显示的是最近的真实行权价与剩余天数,"
                    "而非插值后的数值。\n\n"
                    "典型形态是每个到期日横跨行权价的“微笑”或偏斜(单到期截面见"
                    "“波动率 (IV)”视图),以及随到期日拉远整体 IV 升降(期限结构)。"
                    "曲面上的局部尖刺通常说明该区域数据稀疏,而非真实的市场效应 — "
                    "相信整体形状,别信单个凸起。",
                )
            )

    # THE PRICE CHART LAST, AND ONLY HERE. It used to sit above the view
    # switcher, which meant every one of the nine views paid for an embedded
    # third-party widget — 450 pixels and an outside request — to show the
    # same line, on screens that are about the option chain rather than the
    # underlying. The hosted product keeps it at the foot of Overview for the
    # same reason, and this is where somebody actually looks for it.
    st.subheader(i18n.t("Underlying price", "标的价格"))
    components.html(render_tradingview_widget(selected_ticker, height=450), height=450)


if active_view == "Changes":
    # WHAT THIS VIEW IS FOR. Every other view answers a question about now, and
    # the one thing this tool has that a web page of today's numbers does not
    # is that it kept yesterday. This is the view that says so: two trading
    # days side by side, and what moved between them.
    _days = db.list_day_summaries(conn, selected_ticker, page_source) if page_source else []
    if len(_days) < 1:
        st.info(
            i18n.t(
                "No day summaries yet for this ticker. One is written after every collection, "
                "and the worker fills in the days it missed — so this view fills itself in as "
                "collection runs.",
                "该代码还没有按日汇总。每次采集后都会写一条,worker 会补齐错过的日子 — "
                "随着采集运行,本视图会自动填充。",
            )
        )
    else:
        # The pickers offer days that EXIST rather than a calendar: a date with
        # no collection behind it is not a comparison, and offering it invites
        # somebody to pick it and read an error as a defect.
        _default_after = _days[0]
        _default_before = _days[1] if len(_days) > 1 else _days[0]
        left, right = st.columns(2)
        day_before = left.selectbox(
            i18n.t("Compare from", "比较起点"), _days, index=_days.index(_default_before), key="changes_from",
            format_func=format_date,
        )
        day_after = right.selectbox(
            i18n.t("to", "至"), _days, index=_days.index(_default_after), key="changes_to",
            format_func=format_date,
        )
        row_after = db.get_day_summary(conn, selected_ticker, page_source, day_after)
        row_before = (
            db.get_day_summary(conn, selected_ticker, page_source, day_before)
            if day_before != day_after else None
        )
        if row_after is None:
            st.info(i18n.t("That day has no summary row.", "那一天没有汇总行。"))
        else:
            # The flow numbers live in the day row; the volatility average
            # lives with the collection moment it was computed for. Joined here
            # rather than stored twice.
            def _rollup(row):
                if row is None:
                    return None
                stored = db.get_moment_rollup(
                    conn, selected_ticker, page_source, row["collected_at"]
                ) or {}
                return {
                    "iv_weighted_avg": stored.get("iv_weighted_avg"),
                    "call_volume": row.get("call_volume"), "put_volume": row.get("put_volume"),
                    "call_oi": row.get("call_oi"), "put_oi": row.get("put_oi"),
                }

            rollup_after = _rollup(row_after)
            rollup_before = _rollup(row_before)
            # Yesterday's max pain for TODAY's nearest expiry, on the day the
            # nearest expiry rolls over. Without it the line claims a brand new
            # level, which is the chain being read through a different expiry
            # rather than anything having happened.
            same_expiry_pain = None
            if (
                row_before
                and row_after.get("nearest_expiry")
                and str(row_before.get("nearest_expiry")) != str(row_after.get("nearest_expiry"))
            ):
                earlier = db.get_snapshots_at(
                    conn, selected_ticker, [row_before["collected_at"]], source=page_source
                )
                if not earlier.empty:
                    wanted = pd.Timestamp(row_after["nearest_expiry"])
                    same_day = earlier[
                        pd.to_datetime(earlier["expiry"], errors="coerce") == wanted
                    ]
                    if not same_day.empty:
                        pain = metrics.max_pain(same_day, wanted)
                        same_expiry_pain = None if pain is None or pd.isna(pain) else float(pain)

            ladder = day_summary.changes(
                row_before, row_after, rollup_before, rollup_after, same_expiry_pain
            )
            st.subheader(day_summary.headline(ladder))
            st.caption(
                f"{format_date(day_before)} → {format_date(day_after)}"
                + (
                    i18n.t("  ·  first day on record — nothing to compare yet",
                           "  ·  有记录的第一天 — 尚无可比较对象")
                    if not row_before else ""
                )
            )

            # One HTML table rather than a dataframe: the chips carry meaning
            # in colour (green towards damping, purple towards amplification —
            # what those two colours mean on every other chart here), and a
            # plain table of numbers is exactly the thing this view exists to
            # not be.
            rows_html = []
            for line in ladder:
                tone = ""
                if line["kind"] == day_summary.DAMPING:
                    tone = " gg-green"
                elif line["kind"] == day_summary.AMPLIFYING:
                    tone = " gg-purple"
                note = line.get("note")
                if line["key"] == "expiries":
                    parts = []
                    if line.get("rolled_off"):
                        parts.append("expired: " + ", ".join(line["rolled_off"]))
                    if line.get("listed"):
                        parts.append("newly listed: " + ", ".join(line["listed"]))
                    note = note or "; ".join(parts)
                label = line["label"] + (
                    f'<span class="gg-note">{note}</span>' if note else ""
                )
                rows_html.append(
                    f"<tr><td>{label}</td>"
                    f'<td class="gg-num gg-before">{day_summary.format_value(line, "before")}</td>'
                    f'<td class="gg-num">{day_summary.format_value(line, "after")}</td>'
                    f'<td class="gg-num{tone}">{day_summary.format_delta(line)}</td></tr>'
                )
            st.markdown(
                '<table class="gg-ladder"><tr><th></th>'
                f"<th>{format_date(day_before)}</th>"
                f"<th>{format_date(day_after)}</th>"
                f"<th>{i18n.t('change', '变化')}</th></tr>"
                + "".join(rows_html) + "</table>",
                unsafe_allow_html=True,
            )

            # The two GEX profiles on one chart: the day before as an outline,
            # the day after filled. The window is the strikes around the money,
            # the same band the heatmap draws, because the far tails are where
            # a profile carries nothing and a chart drawn over them shows two
            # flat lines meeting in the middle.
            profile_after = row_after.get("gex_by_strike") or []
            if profile_after:
                st.subheader(i18n.t("GEX profile, both days", "两日 GEX 剖面对比"))
                frame = pd.DataFrame({format_date(day_after): dict(profile_after)})
                if row_before and row_before.get("gex_by_strike"):
                    frame[format_date(day_before)] = pd.Series(
                        dict(row_before["gex_by_strike"])
                    )
                frame = frame.sort_index()
                frame.index.name = "strike"
                frame = strikes_around_money(frame, row_after["underlying_price"], 20)
                # NOT STACKED. Stacked bars would draw one day ON TOP of the
                # other, so a strike where both days carry exposure would read
                # as their sum — a wall twice the size of either day's.
                st.bar_chart(
                    frame, stack=False,
                    color=[BRAND_GREEN, BRAND_PURPLE][: len(frame.columns)],
                )

            with st.expander("ℹ️ How to read this"):
                st.write(
                    i18n.t(
                        "Two trading days of your own collection, side by side. A day is New "
                        "York's date, and the row describes that day's last collection — the "
                        "same rule OI Delta uses, so the two views never disagree about what "
                        "\"yesterday\" means.\n\n"
                        "Colour is reserved for the regime: green when the day moved towards "
                        "dealer hedging damping moves, purple when it moved towards amplifying "
                        "them. Everything else changed without changing what kind of day it "
                        "is, so it is printed in plain ink.\n\n"
                        "Levels are reported in strikes because that is the grid they live on: "
                        "a wall that moved \"+1 strike\" moved to the next listed strike, which "
                        "is a fact about this chain, while the same move as a percentage is a "
                        "fact about nothing. Max pain is compared for the same expiry only — "
                        "when the nearest expiry rolls over, the line says which one was "
                        "nearest before.",
                        "把你自己的采集数据里两个交易日并排对比。“一天”按纽约日期划分,"
                        "每行描述该天最后一次采集 — 与“持仓变化”视图同一规则,"
                        "两个视图对“昨天”的含义永远不会打架。\n\n"
                        "颜色只留给体制变化:绿色代表当天朝“对冲抑制波动”移动,"
                        "紫色代表朝“对冲放大波动”移动。其余变化不改变当天的性质,以普通墨色打印。\n\n"
                        "关键位以档数为单位报告,因为它们本来就长在档位网格上:"
                        "墙“+1 档”就是移到了下一个挂牌行权价,这是这条链本身的事实,"
                        "换算成百分比反而什么都不是。最大痛点只在同一到期日内比较 — "
                        "当最近到期换月时,该行会注明之前最近的是哪一个。",
                    )
                )


if active_view == "Max Pain / GEX":
    # Most tickers list an expiry for today, and it sorts first — so the
    # default landed on the one expiry whose gamma is zero by definition, and
    # the GEX chart came up empty under a "Net GEX: 0" banner while Max Pain
    # right above it showed a number (it needs only open interest). Default to
    # the first expiry that still has time left instead; the others stay in the
    # list, labelled, because looking at them is legitimate.
    has_time_left = [e for e in expiries if metrics.years_to_expiry(e, latest_date) > 0]
    default_expiry_index = expiries.index(has_time_left[0]) if has_time_left else 0

    def format_gex_expiry(expiry) -> str:
        if metrics.years_to_expiry(expiry, latest_date) > 0:
            return format_date(expiry)
        return f"{format_date(expiry)}" + i18n.t("  ⚠️ no gamma left", "  ⚠️ 已无时间价值")

    selected_expiry = st.selectbox(
        i18n.t("Expiry", "到期日"), expiries, index=default_expiry_index, format_func=format_gex_expiry,
        # Keyed by ticker: a widget value outlives a rerun, so without this the
        # expiry picked for the previous ticker carries over, and if the new
        # ticker doesn't trade that date every chart below filters to nothing.
        key=f"expiry_pain_gex_{selected_ticker}",
    )
    years_left = metrics.years_to_expiry(selected_expiry, latest_date)

    st.subheader(i18n.t("Max Pain", "最大痛点"))
    mp = metrics.max_pain(latest_df, selected_expiry)
    st.metric(i18n.t("Max Pain strike", "最大痛点行权价"), mp if mp is not None else "n/a")
    with st.expander(i18n.t("ℹ️ How to read this", "ℹ️ 怎么读这张图")):
        st.write(
            i18n.t(
                "The strike at which total payouts by option sellers are minimal. The price at "
                "expiration statistically gravitates toward this level, since market makers "
                "prefer to close the maximum number of contracts without payouts — the effect "
                "strengthens closer to the expiry date. It is not a guaranteed forecast but a "
                "tendency that fundamental news can override.",
                "期权卖方总赔付最小的行权价。到期日价格在统计上会向该水平靠拢,"
                "因为做市商倾向于让尽可能多的合约过期作废、无需赔付 — 越临近到期该效应越强。"
                "它不是必然的预测,只是一个会被基本面消息覆盖的倾向。",
            )
        )

    st.subheader(i18n.t("Approximate Gamma Exposure (GEX)", "近似 Gamma 敞口(GEX)"))
    gex = metrics.gamma_exposure_profile(latest_df, selected_expiry)
    if years_left <= 0:
        # Say why, instead of drawing a flat line at zero. Gamma is undefined
        # once there is no time left, so this is a property of the contract,
        # not a gap in the data — which is also why Max Pain above still shows
        # a number: it is computed from open interest alone.
        st.info(
            i18n.t(
                f"⚠️ This expiry ({format_date(selected_expiry)}) has no time left as of the "
                f"latest collection ({format_datetime(latest_date)}), so gamma — and therefore "
                "GEX — is zero by definition, not missing. Pick a later expiry to see a profile.",
                f"⚠️ 截至最近一次采集({format_datetime(latest_date)}),该到期日"
                f"({format_date(selected_expiry)})已无剩余时间,因此 Gamma(以及 GEX)"
                "按定义为零,不是数据缺失。选择更晚的到期日可看到轮廓。",
            )
        )
    elif not gex.empty:
        net_gex = metrics.net_gamma_exposure(gex)
        if net_gex >= 0:
            st.success(i18n.t(f"Net GEX: {net_gex:,.0f} — positive gamma: dealers dampen price moves",
                              f"净 GEX:{net_gex:,.0f} — 正 Gamma:做市商对冲抑制价格波动"))
        else:
            st.warning(i18n.t(f"Net GEX: {net_gex:,.0f} — negative gamma: dealers amplify price moves",
                              f"净 GEX:{net_gex:,.0f} — 负 Gamma:做市商对冲放大价格波动"))
        # Colored by sign (green/purple), matching the same diverging language
        # as the GEX Heatmap — st.bar_chart can't condition color on a row's
        # own value, so this one chart uses Altair directly instead of the
        # thin st.bar_chart wrapper.
        _pos, _neg = i18n.t("Positive", "正 GEX"), i18n.t("Negative", "负 GEX")
        gex_chart_data = gex.assign(sign=np.where(gex["gex"] >= 0, _pos, _neg))
        st.altair_chart(
            alt.Chart(gex_chart_data)
            .mark_bar()
            .encode(
                x=alt.X("strike:O", title=i18n.t("Strike", "行权价")),
                y=alt.Y("gex:Q", title="GEX"),
                color=alt.Color(
                    "sign:N",
                    scale=alt.Scale(domain=[_pos, _neg], range=[BRAND_GREEN, BRAND_PURPLE]),
                    legend=None,
                ),
            ),
            use_container_width=True,
        )
    else:
        st.info(i18n.t("No data to compute GEX for the selected expiry.",
                       "所选到期日没有可计算 GEX 的数据。"))
    st.caption(
        i18n.t(
            "Estimated via the Black-Scholes formula and a sign heuristic (puts contribute "
            "negatively). Does not reflect actual market-maker positioning — an approximation only.",
            "由 Black-Scholes 公式与符号启发式估计(Put 记为负)。不代表做市商真实持仓 — 仅为近似。",
        )
    )
    with st.expander(i18n.t("ℹ️ How to read this", "ℹ️ 怎么读这张图")):
        st.write(
            i18n.t(
                "Positive total gamma — the market as a whole is more \"sticky\", range-bound: "
                "dealer hedging dampens sharp moves. Negative — moves can accelerate, since "
                "hedging works in the same direction as the trend. On the by-strike chart, the "
                "largest bars by magnitude often become support/resistance levels created by "
                "hedging flows, especially closer to the expiry date.",
                "总 Gamma 为正 — 市场整体更“粘”,倾向区间震荡:做市商对冲会抑制剧烈波动。"
                "为负 — 波动可能加速,因为对冲方向与趋势同向。在按行权价的柱状图上,"
                "数值最大的几档常常形成由对冲流制造的支撑/阻力,越临近到期越明显。",
            )
        )

if active_view == "GEX Heatmap":
    st.subheader(i18n.t("GEX Heatmap: strike × expiry", "GEX 热力图:行权价 × 到期日"))

    # From the run log rather than DISTINCT over the snapshots: the collector
    # writes one timestamp per pass into both, and the log has one row per pass
    # against one row per contract per pass.
    snapshot_dates = [
        pd.Timestamp(moment)
        for moment in db.get_collection_moments(conn, selected_ticker, days=history_days, source=page_source)
    ] or [pd.Timestamp(latest_date)]
    as_of = st.selectbox(
        i18n.t("Snapshot (Replay — collection history, no auto-refresh)",
               "快照(回放 — 采集历史,不自动刷新)"),
        snapshot_dates,
        format_func=format_datetime,
        key="heatmap_as_of",
    )
    is_latest = as_of == snapshot_dates[0]
    st.caption(i18n.t("Latest snapshot (current state)", "最新快照(当前状态)")
               if is_latest else i18n.t("Historical snapshot — Replay mode", "历史快照 — 回放模式"))

    # The chosen moment's chain, fetched by equality on collected_at. Everything
    # this view draws — the matrix, the walls, the flip, per-expiry net GEX — is
    # a function of one moment, and every metric below already takes `as_of`.
    snapshot_df = with_our_iv(
        db.get_snapshots_at(conn, selected_ticker, [as_of], source=page_source)
    )
    if snapshot_df.empty:
        # The run log is a superset of the data: a pass can be recorded as a
        # success and still have left nothing for this ticker. Before the list
        # came from the log, `as_of` came from the rows themselves and could
        # not miss.
        st.info(i18n.t("That collection is in the log but stored no rows — pick another moment.",
                       "该次采集在日志中但没有存储任何行 — 请换一个时刻。"))
        st.stop()

    # Expiries with no time left are dropped rather than shown as a column of
    # exact zeros: gamma is undefined there, and unlike the GEX tab a heatmap
    # has nowhere to explain why one column is blank. Deliberately NOT st.stop()
    # when none are left — this runs inside `with tab_heatmap:`, and st.stop()
    # ends the entire script run, taking every tab below it with it.
    all_expiries_at_snapshot = [
        e for e in sorted(snapshot_df["expiry"].unique())
        if metrics.years_to_expiry(e, as_of) > 0
    ]
    spot_at_snapshot = snapshot_df["underlying_price"].iloc[0]

    if not all_expiries_at_snapshot:
        st.info(
            i18n.t(
                "Every expiry in this snapshot has already stopped trading, so the whole "
                "matrix would be zero by definition. Pick a more recent snapshot above.",
                "该快照中所有到期日都已停止交易,整张矩阵按定义全为零。请在上方选择更新的快照。",
            )
        )
        matrix_full = pd.DataFrame()
        shown_expiries = []
        strikes_each_side = 10
    else:
        col_n, col_band = st.columns(2)
        # A slider needs min < max: with exactly one expiry to show, st.slider
        # raises instead of rendering a fixed handle. Found by rendering the
        # page for a ticker that trades a single expiry — thin tickers really
        # do look like this, and it took out the whole tab.
        if len(all_expiries_at_snapshot) > 1:
            n_expiries = col_n.slider(
                i18n.t("Nearest expiries", "最近 N 个到期日"),
                1,
                len(all_expiries_at_snapshot),
                min(10, len(all_expiries_at_snapshot)),
                # Keyed by ticker: the upper bound is the number of expiries,
                # which differs per ticker, so a carried-over value can fall
                # outside the range and raise rather than merely render oddly.
                key=f"heatmap_n_expiries_{selected_ticker}",
            )
        else:
            n_expiries = 1
            col_n.caption(i18n.t("Nearest expiries: 1 (the only one still trading)",
                                 "最近到期:1 个(唯一还在交易的)"))
        # COUNTED IN STRIKES, NOT IN PERCENT. The
        # control asked for ±% until this day, and a percentage is a different
        # request on every chain: ±5% is 8 strikes on MO, 77 on SPY, 154 on SPX
        # — dense enough that the 45-row window below it silently overrode
        # whatever the slider said, on the two most-watched symbols there are.
        # A count means the same thing on every chain and keeps the money in
        # the middle of the table by construction.
        strikes_each_side = col_band.slider(
            i18n.t("Strikes each side of the money", "平值两侧各取档数"), 5, 100, 10, key="heatmap_strikes_each_side",
            help=(
                i18n.t(
                    "How many strikes above and below the strike closest to the underlying "
                    "price to draw. 10 shows 21 rows.",
                    "在最接近标的价格的行权价上下各绘制多少档。10 档显示 21 行。",
                )
            ),
        )
        shown_expiries = all_expiries_at_snapshot[:n_expiries]
        matrix_full = metrics.gex_matrix(snapshot_df, as_of=as_of, expiries=shown_expiries)

    if matrix_full.empty:
        st.info(i18n.t("No data to build the heatmap for the selected snapshot.",
                       "所选快照没有可用于构建热力图的数据。"))
    else:
        matrix_band = strikes_around_money(matrix_full, spot_at_snapshot, strikes_each_side)

        walls = metrics.dealer_walls(snapshot_df, as_of=as_of, expiries=shown_expiries)
        # BOTH READ OFF THE MATRIX ABOVE instead of pricing the chain again.
        # `gamma_flip_price` and `net_gex_by_expiry` each rebuilt it
        # internally, so this screen evaluated the same greeks three times over
        # — and every slider on it reruns the whole script, so the cost was
        # paid on each nudge. Measured on the hosted product, whose matrix is
        # the same function: 854.6 ms to 66.7 ms on a 13,160-contract SPY
        # chain, and the numbers coming out are bit-for-bit the ones that went
        # in before (max |Δ| = 0 over 357 × 10 cells of a live chain). The flip
        # still walks the full strike range rather than the displayed band —
        # `matrix_full` is the unbanded matrix.
        flip = metrics.gamma_flip_from_matrix(matrix_full, spot_at_snapshot)
        net_by_expiry = metrics.net_gex_from_matrix(matrix_full, expiries=shown_expiries)
        total_net_gex = net_by_expiry["net_gex"].sum()

        col_price, col_call, col_put, col_flip, col_zone = st.columns(5)
        col_price.metric(i18n.t("Underlying price", "标的价格"), f"{spot_at_snapshot:,.2f}")
        col_call.metric(i18n.t("Call Wall", "Call 墙"), f"{walls['call_wall']:g}" if walls["call_wall"] is not None else "n/a")
        col_put.metric(i18n.t("Put Wall", "Put 墙"), f"{walls['put_wall']:g}" if walls["put_wall"] is not None else "n/a")
        col_flip.metric(i18n.t("Gamma Flip", "Gamma 翻转"), f"{flip:,.2f}" if flip is not None else "n/a")
        col_zone.metric(i18n.t("Regime", "体制"), i18n.t("Neg", "负") if total_net_gex < 0 else i18n.t("Pos", "正"))

        if matrix_band.empty:
            st.info(i18n.t("No strikes to show for this snapshot.", "该快照没有可显示的行权价档位。"))
        else:
            # THE CONTROL DECIDES WHAT IS DRAWN. What used to be
            # here was a 45-row window around the money, applied AFTER the
            # control had chosen a percentage band — and it quietly overrode it:
            # SPY has 77 strikes within ±5% and SPX has 154, so on both, the two
            # most watched symbols in the product, the slider produced the
            # identical table at every position from 5% to 50%. A control that
            # silently ignores its input is worse than no control.
            #
            # The count above now IS the window, so there is nothing left to
            # override it, and the money stays in the middle by construction.
            strikes = matrix_band.index.to_numpy(dtype=float)
            atm_strike = matrix_band.index[np.abs(strikes - spot_at_snapshot).argmin()]
            matrix = matrix_band

            # Expiries that are all zeros within the shown strike range get
            # dropped: a column without a single nonzero value carries no
            # information and just widens the table to the right.
            nonzero_cols = matrix.columns[(matrix != 0).any(axis=0)]
            hidden_count = len(matrix.columns) - len(nonzero_cols)
            matrix = matrix[nonzero_cols]
            net_by_expiry_shown = net_by_expiry[net_by_expiry["expiry"].isin(nonzero_cols)]

            st.caption(
                i18n.t(
                    f"Net GEX across {len(nonzero_cols)} of {len(shown_expiries)} shown expiries "
                    f"({len(all_expiries_at_snapshot)} total)"
                    + (f" — {hidden_count} hidden (no activity in this strike range)" if hidden_count else "")
                    + ":",
                    f"净 GEX 覆盖显示的 {len(shown_expiries)} 个到期中的 {len(nonzero_cols)} 个"
                    f"(共 {len(all_expiries_at_snapshot)} 个)"
                    + (f" — 隐藏 {hidden_count} 个(该行权价区间内无持仓)" if hidden_count else "")
                    + ":",
                )
            )
            net_display = net_by_expiry_shown.assign(
                expiry=net_by_expiry_shown["expiry"].map(format_date),
                net_gex=net_by_expiry_shown["net_gex"].map(format_compact),
            )
            st.dataframe(net_display.set_index("expiry").T, use_container_width=True)

            # A viewport, not a filter — the difference this fix is
            # about. 45 rows is a comfortable screenful; asking for more than
            # that many strikes costs scrolling inside the table rather than a
            # page several thousand pixels long.
            viewport_rows = 45
            st.caption(
                i18n.t(
                    f"{len(matrix)} strikes around the money ({atm_strike:g}, marked “➤”): "
                    f"up to {strikes_each_side} each side"
                    + (" — the table scrolls" if len(matrix) > viewport_rows else ""),
                    f"平值附近 {len(matrix)} 个行权价({atm_strike:g},以“➤”标出):"
                    f"两侧各最多 {strikes_each_side} 档"
                    + (" — 表格可滚动" if len(matrix) > viewport_rows else ""),
                )
            )
            # Compact row height — without it only ~10 rows fit on screen, and
            # comparing all strikes required constant scrolling.
            row_height = 24
            table_height = (min(len(matrix), viewport_rows) + 1) * row_height + 3
            st.dataframe(
                style_gex_matrix(matrix, atm_strike=atm_strike),
                use_container_width=True,
                height=table_height,
                row_height=row_height,
            )

        with st.expander(i18n.t("ℹ️ How to read this", "ℹ️ 怎么读这张图")):
            st.write(
                i18n.t(
                    "Each cell is the approximate GEX (Black-Scholes, same sign heuristic as on the "
                    "Max Pain/GEX tab) for a specific strike and expiry, computed on a single "
                    "snapshot — no need to pick expiries one at a time, the whole chain is visible at "
                    "once. Numbers are abbreviated (1.2k = 1,200, 1.5m = 1,500,000) — exact values "
                    "aren't needed for visual comparison, and shorter text leaves more room for "
                    "expiry columns. Green — positive GEX (dealers dampen the move), purple — "
                    "negative (dealers amplify the move), black — near zero. The color scale is "
                    "calibrated to the 90th percentile of values rather than the true maximum — "
                    "otherwise a single strong outlier stretches the scale until every other cell "
                    "becomes nearly indistinguishably washed out; the most extreme values are simply "
                    "painted the limit color, and the exact number is still visible as text. 0 means "
                    "this strike simply has no listing for that expiry (not \"no data\") — that's "
                    "normal: near expiries usually have a tighter range of actual strikes than far "
                    "ones; expiries with no nonzero values at all within the shown strike range are "
                    "hidden from the table entirely. The strike closest to the current underlying "
                    "price is marked “➤” and outlined, and the table shows a strike window "
                    "centered on it.\n\n"
                    "**Call Wall / Put Wall** — the strike with maximum open interest in calls/puts "
                    "respectively, aggregated over the shown expiries — a common proxy for "
                    "support/resistance levels created by dealer hedging flows.\n\n"
                    "**Gamma Flip** — the approximate underlying price at which total GEX across the "
                    "shown expiries flips sign (linear interpolation between the nearest strikes). "
                    "This is not a re-pricing of greeks at hypothetical underlying prices (more "
                    "accurate but noticeably more expensive to compute), but a proxy based on the "
                    "already-computed profile at the current price — the same class of assumption as "
                    "net GEX. **Regime** Neg/Pos — the sign of total net GEX across the shown "
                    "expiries.\n\n"
                    "A ticker can have 20-30+ expiries (including far-dated LEAPS 1-2 years out) "
                    "with very different strike ranges — the full matrix would be mostly empty and "
                    "unreadable. The sliders above limit the view to the nearest N expiries and a "
                    "number of strikes each side of the money; the expiry count also decides which "
                    "subset Call/Put Wall, Gamma Flip, and per-expiry Net GEX are computed over, "
                    "while the strike count is display only — the flip is found over the whole "
                    "strike range. The approximation formulas themselves don't change.\n\n"
                    "The snapshot selector lets you pick an earlier collection from history (Replay) "
                    "and see how the picture looked at that moment — no new data is needed, and "
                    "there is no real-time auto-refresh (collection happens only via the “Collect "
                    "data” button).",
                    "每个格子是某个行权价×某个到期日的近似 GEX(Black-Scholes,与“最大痛点 / GEX”"
                    "视图同一套符号约定),基于单次快照计算 — 无需逐个到期日切换,整条链一眼可见。"
                    "数字做了缩写(1.2k = 1,200,1.5m = 1,500,000)— 视觉对比不需要精确值,"
                    "更短的文本给到期列留出更多空间。绿色 — 正 GEX(做市商抑制波动),"
                    "紫色 — 负 GEX(做市商放大波动),黑色 — 接近零。色标按数值的 90 分位校准,"
                    "而非真实最大值 — 否则一个强离群值会把色标拉到其他格子全部难以分辨;"
                    "超出范围的极值直接涂成端点色,精确数字仍以文本显示。0 表示该行权价在"
                    "该到期日根本没有挂牌(不是“没数据”)— 这很正常:近月到期的真实行权价"
                    "覆盖范围通常比远月窄;在所示行权价区间内完全无非零值的到期列会整列隐藏。"
                    "最接近当前标的价格的行权价以“➤”标出并高亮,表格显示以它为中心的窗口。\n\n"
                    "**Call 墙 / Put 墙** — 在所示到期范围内,Call/Put 持仓量最大的那一档 — "
                    "常被用作做市商对冲流制造的支撑/阻力参考位。\n\n"
                    "**Gamma 翻转** — 所示到期的总 GEX 变号的近似标的价格"
                    "(相邻行权价之间线性插值)。这不是在假设价格下重新定价 Greeks"
                    "(更精确但计算贵得多),而是基于当前价格已算出的剖面做代理 — "
                    "与净 GEX 同一类的近似假设。**体制** 负/正 — 所示到期总净 GEX 的符号。\n\n"
                    "一个代码可以有 20-30+ 个到期日(包括一两年后的远期 LEAPS),"
                    "行权价范围差异极大 — 完整矩阵会大片空白、没法读。上方的滑块把视图限制在"
                    "最近 N 个到期和价差两侧若干档;到期数量同时决定 Call/Put 墙、Gamma 翻转、"
                    "分到期净 GEX 的计算范围,而行权价档数只影响显示 — 翻转位始终在全档位范围"
                    "内寻找。近似公式本身不变。\n\n"
                    "快照选择器可以挑选历史上的某次采集(回放),看看那一刻的图景 — "
                    "不需要新数据,也没有实时自动刷新(采集只通过“立即采集”按钮或定时任务进行)。",
                )
            )

if active_view == "Volatility (IV)":
    st.subheader(i18n.t("IV: volume-weighted average for the ticker", "IV:该代码的成交量加权均值"))
    # From the rollup written at collection time, not aggregated on read: the
    # answer is one row per collection and the input was one row per contract
    # per collection. It also stops the chart changing under you as contracts
    # expire and move to the archive.
    iv_avg = db.get_iv_weighted_average(conn, selected_ticker, days=history_days, source=page_source)
    st.line_chart(with_viewer_index(iv_avg)["iv_weighted_avg"], color=BRAND_PURPLE)
    # THE ONE PLACE A READER COULD SEE TWO MODELS IN ONE PICTURE. Everything
    # else on this page is solved as it is drawn, so it is ours all the way
    # back; this chart is read from a rollup, and every point written before
    # the release that changed the model holds the data source's number. The
    # collector rewrites them in the background, newest first, so the line says
    # what is left rather than warning about something the reader cannot act on
    # — and it disappears by itself when there is nothing left to say.
    pending_moments = db.iv_summary_pending(conn, selected_ticker, source=page_source)
    if pending_moments:
        st.caption(
            i18n.t(
                f"The earliest {pending_moments:,} point(s) of this chart still carry the data "
                "source's own volatility, from before this product started solving it from the "
                "contract's price. They are being recomputed in the background while the "
                "collector runs — nothing to start, and the rest of this page is unaffected.",
                f"本图最早的 {pending_moments:,} 个点仍带着数据源自带的波动率"
                "(本产品开始用合约价格反解之前的历史)。采集器运行时会在后台重算 — "
                "无需任何操作,页面其余部分不受影响。",
            )
        )
    with st.expander(i18n.t("ℹ️ How to read this", "ℹ️ 怎么读这张图")):
        st.write(
            i18n.t(
                "Rising average IV usually precedes an anticipated move (earnings, news) or "
                "already reflects increased market uncertainty. Falling — the market is calming "
                "down. Compare the current level against the historical range on this chart, "
                "not against absolute numbers — \"normal\" IV differs a lot between tickers.",
                "平均 IV 上升通常预示市场在为即将到来的波动定价(财报、消息),"
                "或已经反映了不确定性的抬升;下降则代表市场在平静下来。"
                "把当前水平和本图里的历史区间比,而不是和绝对数字比 — "
                "“正常”的 IV 水平在不同代码之间差别巨大。",
            )
        )

    st.subheader(i18n.t("IV: chain slice (latest snapshot)", "IV:链截面(最新快照)"))
    selected_expiry_iv = st.selectbox(
        i18n.t("Expiry", "到期日"), expiries, format_func=format_date, key=f"expiry_iv_skew_{selected_ticker}"
    )
    skew_snapshot = latest_df[latest_df["expiry"] == selected_expiry_iv]
    skew_pivot = skew_snapshot.pivot_table(
        index="strike", columns="option_type", values="implied_volatility"
    )
    if skew_pivot.empty:
        st.info(i18n.t("No data for this expiry in the latest snapshot.",
                       "最新快照中没有该到期日的数据。"))
    else:
        # A fixed 2-color list crashes (StreamlitColorLengthError, found live
        # on real illiquid tickers) whenever the latest snapshot for this
        # expiry only has calls or only puts — pivot_table then produces a
        # single column, not two. Built to match skew_pivot's actual columns
        # instead of assuming both are always present.
        option_type_colors = {"call": BRAND_PURPLE, "put": BRAND_GREEN}
        st.line_chart(skew_pivot, color=[option_type_colors[c] for c in skew_pivot.columns])
    with st.expander(i18n.t("ℹ️ How to read this", "ℹ️ 怎么读这张图")):
        st.write(
            i18n.t(
                "The curve's shape shows which strikes the market prices as riskier (higher "
                "implied volatility). A steep skew toward puts at low strikes (put skew) "
                "usually means elevated demand for downside protection — the typical picture "
                "for most stocks.",
                "曲线的形状显示市场认为哪些行权价风险更高(隐含波动率更高)。"
                "低行权价一端 Put IV 明显抬高(Put 偏斜)通常意味着下行保护需求旺盛 — "
                "这也是多数股票的常态。",
            )
        )

    st.caption(i18n.t("IV history and the rest of the greeks for a specific contract — on the Contract tab.",
                      "单个合约的 IV 历史与其余 Greeks — 见“单合约”视图。"))

if active_view == "Contract":
    st.subheader(i18n.t("Contract: price and greeks over time", "合约:价格与 Greeks 时间序列"))

    if not get_tracked().empty:
        st.caption(i18n.t("Pinned contracts (click a name to show it below):",
                          "已固定的合约(点击名称在下方展示):"))
        for _, trow in get_tracked().iterrows():
            col_a, col_b = st.columns([5, 1])
            label = f"{format_date(trow['expiry'])}  strike {trow['strike']:g}  {trow['option_type']}"
            if col_a.button(label, key=f"select_tracked_{trow['id']}", use_container_width=True):
                # Written before the rerun so the selectors below initialize
                # with these values. The keys MUST match the ones the selectors
                # use, ticker suffix included — they were keyed by ticker to stop
                # one ticker's expiry leaking into another, and this handler kept
                # writing the old unsuffixed names, so clicking a pinned contract
                # silently did nothing at all.
                st.session_state[f"opt_expiry_{selected_ticker}"] = trow["expiry"]
                st.session_state[f"opt_strike_{selected_ticker}"] = trow["strike"]
                st.session_state[f"opt_type_{selected_ticker}"] = trow["option_type"]
                st.rerun()
            if col_b.button("✕", key=f"untrack_{trow['id']}"):
                db.remove_tracked_contract(conn, trow["id"])
                st.rerun()
        st.divider()

    # EXPIRED CONTRACTS. Their history is stored and perfectly readable; what
    # they cannot do is appear in these selectors, because the lists are built
    # from the LATEST snapshot and it has no row for something that no longer
    # trades. On expiry day a contract simply vanishes from the dropdown — and
    # if a stale expiry survives in the widget's state, the strike list under it
    # comes back empty and the page shows a validly selected contract with no
    # data at all.
    #
    # Off by default: on this database 7,098 of 26,894 contracts have already
    # expired, a ratio that only grows, and always-on would bury the live chain
    # under the archive.
    expired = db.get_expired_contracts(conn, selected_ticker, source=page_source)
    expired_expiries = []
    if not expired.empty:
        if st.checkbox(
            i18n.t(f"Show expired contracts ({len(expired)} with collected history)",
                   f"显示已到期合约({len(expired)} 个有已采集历史)"),
            key=f"show_expired_{selected_ticker}",
            help=i18n.t(
                "Contracts that have already expired. Nothing was deleted — they simply "
                "cannot appear in these lists on their own, because the lists come from "
                "the current chain.",
                "已到期的合约。没有任何数据被删除 — 只是这些列表来自当前未到期链,"
                "它们无法自行出现在列表里。",
            ),
        ):
            expired_expiries = sorted(pd.Timestamp(e) for e in expired["expiry"].unique())

    live_expiries = [pd.Timestamp(e) for e in expiries]
    all_expiries = sorted(set(live_expiries) | set(expired_expiries))

    def _format_expiry_option(expiry) -> str:
        label = format_date(expiry)
        return label if expiry in live_expiries else label + i18n.t("  ⏳ expired", "  ⏳ 已到期")

    col1, col2, col3 = st.columns(3)
    # All three keyed by ticker for the same reason as the GEX expiry above —
    # and the strike list especially: strikes differ between tickers, so a
    # carried-over value can be one this ticker does not trade at all.
    _drop_stale_choice(f"opt_expiry_{selected_ticker}", all_expiries)
    opt_expiry = col1.selectbox(
        i18n.t("Expiry", "到期日"), all_expiries, format_func=_format_expiry_option,
        key=f"opt_expiry_{selected_ticker}",
    )
    live_strikes = set(latest_df[latest_df["expiry"] == opt_expiry]["strike"].unique())
    expired_strikes = set()
    if expired_expiries:
        expired_strikes = set(expired[expired["expiry"] == opt_expiry]["strike"].unique())
    opt_strikes = sorted(live_strikes | expired_strikes)
    # A stored widget value that is no longer among the options makes Streamlit
    # raise, and the option lists here narrow as you go: strikes depend on the
    # expiry, and the sides quoted depend on the strike. Drop the stale value
    # rather than letting the tab die on a legitimate sequence of clicks.
    _drop_stale_choice(f"opt_strike_{selected_ticker}", opt_strikes)
    opt_strike = col2.selectbox(i18n.t("Strike", "行权价"), opt_strikes, key=f"opt_strike_{selected_ticker}")
    # Only the sides actually quoted for this expiry and strike. Offering both
    # unconditionally let you select a contract that was never collected — far
    # OTM strikes are routinely listed on one side only — and the app then said
    # "No history for the selected contract", which is true and useless.
    available_types = sorted(
        latest_df[(latest_df["expiry"] == opt_expiry) & (latest_df["strike"] == opt_strike)]["option_type"].unique()
    )
    if not available_types and expired_expiries:
        available_types = sorted(
            expired[(expired["expiry"] == opt_expiry) & (expired["strike"] == opt_strike)]["option_type"].unique()
        )
    available_types = available_types or ["call", "put"]
    _drop_stale_choice(f"opt_type_{selected_ticker}", available_types)
    opt_type = col3.selectbox(i18n.t("Type", "类型"), available_types, key=f"opt_type_{selected_ticker}")

    # THE WINDOW HAS TO COME OFF FOR AN EXPIRED CONTRACT. The day limit is
    # measured back from now(), and an expired contract's history is entirely in
    # the past — so the limit trims it silently, and once the contract is older
    # than the limit it trims it to nothing. The page would then say "no history"
    # about a contract whose history is sitting in the table.
    is_expired = pd.Timestamp(opt_expiry) < pd.Timestamp(dt.date.today())
    contract_df = with_our_iv(db.get_contract_history(
        conn, selected_ticker, opt_expiry, opt_strike, opt_type,
        days=None if is_expired else history_days, source=page_source,
    ))
    if is_expired and not contract_df.empty:
        first, last = contract_df["collected_at"].min(), contract_df["collected_at"].max()
        st.caption(
            i18n.t(
                f"⏳ **Expired contract** — expired {format_date(opt_expiry)}. Stored history "
                f"covers {format_date(first)} to {format_date(last)}, which starts when this "
                "ticker was first collected rather than when the contract was listed.",
                f"⏳ **已到期合约** — 于 {format_date(opt_expiry)} 到期。已存储历史覆盖 "
                f"{format_date(first)} 至 {format_date(last)},起点是该代码首次采集的时间,"
                "而非合约挂牌时间。",
            )
        )
    render_option_detail(
        conn, contract_df, selected_ticker, get_tracked(), opt_expiry, opt_strike, opt_type,
        key_prefix="opt",
    )

if active_view == "Screener":
    st.subheader(i18n.t(f"Options screener: {selected_ticker}", f"期权筛选器:{selected_ticker}"))
    st.caption(
        i18n.t(
            "A slice of the latest snapshot (not history) — every contract of the ticker with "
            "greeks and DTE. Clicking a row opens the same charts as the Contract tab.",
            "最新快照的一个切片(非历史)— 该代码的每个合约及其 Greeks 与剩余天数。"
            "点击行会打开与“单合约”视图相同的图表。",
        )
    )

    screener = metrics.screener_table(latest_df)
    # The greeks and DTE are correct AS OF THE SNAPSHOT — that is the only
    # honest way to price them. But collection here is a manual button, so the
    # latest snapshot can be days or weeks old, and contracts that were live
    # when it was taken may have expired since. Showing them in a "what can I
    # trade" screener with a cheerful positive DTE is misleading, so they are
    # hidden by default rather than silently mixed in — and counted out loud,
    # with a toggle, because hiding collected data without saying so is worse
    # than showing it.
    # years_to_expiry normalizes a tz-aware "now" itself — that is what it is
    # for, and doing the conversion at the call site is how the tz-aware/naive
    # mismatch keeps coming back.
    now = pd.Timestamp.utcnow()
    expired_since = pd.Series(dtype=bool)
    if not screener.empty:
        expired_since = screener["expiry"].map(lambda e: metrics.years_to_expiry(e, now) <= 0)
    show_expired = False
    if expired_since.any():
        stale_days = int(metrics.years_to_expiry(latest_date, now) * -365)
        show_expired = st.checkbox(
            i18n.t(
                f"Include {int(expired_since.sum())} contract(s) that have expired since this "
                f"snapshot was collected ({stale_days} day(s) ago)",
                f"包含自该快照采集以来已到期的 {int(expired_since.sum())} 个合约"
                f"({stale_days} 天前)",
            ),
            key=f"screener_show_expired_{selected_ticker}",
            help=i18n.t(
                "Their greeks and DTE are as of the collection, which is why they can "
                "still show time left. Collect again to refresh the chain.",
                "其 Greeks 与剩余天数是采集时刻的数值,所以仍显示有剩余时间。重新采集即可刷新链。",
            ),
        )
    if not show_expired and not screener.empty:
        screener = screener[~expired_since].reset_index(drop=True)

    if screener.empty:
        st.info(i18n.t("No data for the screener.", "筛选器没有数据。"))
    else:
        FILTER_SPECS = [
            ("last_price", i18n.t("Option price", "期权价格")),
            ("strike", i18n.t("Strike", "行权价")),
            ("dte", i18n.t("DTE (days)", "剩余天数")),
            ("open_interest", i18n.t("Open interest", "持仓量")),
            ("implied_volatility", "IV"),
            ("delta", "Delta"),
            ("gamma", "Gamma"),
            ("theta", "Theta"),
            ("vega", "Vega"),
            ("rho", "Rho"),
            ("vanna", "Vanna"),
            ("charm", "Charm"),
        ]
        # Number inputs instead of sliders: some columns (gamma, vega, IV at far
        # strikes, etc.) have a tiny value range, and a slider can't hit it
        # precisely — exact numeric input removes that problem outright. Min/max
        # come from the actual data (full range = nothing filtered by default).
        INT_FILTER_COLUMNS = {"dte", "open_interest"}
        with st.expander(i18n.t("🔍 Filters (ranges)", "🔍 筛选(区间)"), expanded=True):
            filter_cols = st.columns(3)
            mask = pd.Series(True, index=screener.index)
            for i, (col_name, label) in enumerate(FILTER_SPECS):
                series = screener[col_name].dropna()
                col_min, col_max = float(series.min()), float(series.max())
                target = filter_cols[i % 3]
                if col_min == col_max:
                    target.caption(i18n.t(f"{label}: {col_min:g} (single value)",
                                          f"{label}: {col_min:g}(单一值)"))
                    continue
                target.caption(label)
                sub_left, sub_right = target.columns(2)
                # Widget keys include the ticker: without this, when switching
                # tickers Streamlit keeps the previous filter values under the
                # same key (value= only applies at first widget mount) — the
                # bounds silently stay from the previous ticker, which either
                # hides part of the new ticker's contracts or (if the old range
                # no longer intersects the new data) empties the table entirely.
                _from_lbl, _to_lbl = i18n.t("from", "从"), i18n.t("to", "到")
                if col_name in INT_FILTER_COLUMNS:
                    col_min, col_max = int(col_min), int(col_max)
                    selected_min = sub_left.number_input(
                        _from_lbl, min_value=col_min, max_value=col_max, value=col_min,
                        step=1, key=f"screener_filter_{selected_ticker}_{col_name}_min",
                    )
                    selected_max = sub_right.number_input(
                        _to_lbl, min_value=col_min, max_value=col_max, value=col_max,
                        step=1, key=f"screener_filter_{selected_ticker}_{col_name}_max",
                    )
                else:
                    step = max((col_max - col_min) / 100, 1e-6)
                    selected_min = sub_left.number_input(
                        _from_lbl, min_value=col_min, max_value=col_max, value=col_min,
                        step=step, format="%.4f", key=f"screener_filter_{selected_ticker}_{col_name}_min",
                    )
                    selected_max = sub_right.number_input(
                        _to_lbl, min_value=col_min, max_value=col_max, value=col_max,
                        step=step, format="%.4f", key=f"screener_filter_{selected_ticker}_{col_name}_max",
                    )
                mask &= screener[col_name].between(selected_min, selected_max)

        filtered = screener[mask].reset_index(drop=True)
        st.caption(i18n.t(f"Contracts shown: {len(filtered)} of {len(screener)}",
                          f"显示合约:{len(filtered)} / {len(screener)}"))

        selected_contract_key = f"screener_selected_contract_{selected_ticker}"
        scroll_target_key = f"screener_scroll_target_{selected_ticker}"
        selected_contract = st.session_state.get(selected_contract_key)
        if filtered.empty:
            st.info(i18n.t("No contract falls within the selected filter ranges.",
                           "没有合约落在所选筛选区间内。"))
        else:
            # A custom checkbox column instead of st.dataframe's built-in check
            # column: this way it gets a label, and the remaining columns can be
            # made disabled so they can't be accidentally "selected"/opened in
            # the edit overlay — only this column is interactive. dtype=bool is
            # explicit: on an empty list pandas would default to float64, and a
            # CheckboxColumn over a float column crashes with
            # StreamlitAPIException.
            display = filtered.copy()
            display["expiry"] = display["expiry"].map(format_date)
            _show_lbl = i18n.t("Show details", "查看详情")
            display.insert(0, _show_lbl, pd.array([
                (row.expiry, row.strike, row.option_type) == selected_contract
                for row in filtered.itertuples()
            ], dtype=bool))

            edited = st.data_editor(
                display,
                use_container_width=True,
                hide_index=True,
                disabled=[c for c in display.columns if c != _show_lbl],
                column_config={_show_lbl: st.column_config.CheckboxColumn(_show_lbl)},
                key=f"screener_table_{selected_ticker}",
            )

            checked = filtered[edited[_show_lbl]]
            if checked.empty:
                new_contract = None
            else:
                # if more than one row somehow ends up checked — the last
                # checked one wins (single-select emulation over checkboxes)
                picked_row = checked.iloc[-1]
                new_contract = (picked_row["expiry"], picked_row["strike"], picked_row["option_type"])

            if new_contract != selected_contract:
                st.session_state[selected_contract_key] = new_contract
                st.rerun()

        if selected_contract is not None:
            picked = filtered[
                (filtered["expiry"] == selected_contract[0])
                & (filtered["strike"] == selected_contract[1])
                & (filtered["option_type"] == selected_contract[2])
            ]
            if not picked.empty:
                picked = picked.iloc[0]
                st.divider()
                st.subheader(
                    i18n.t(
                        f"📌 Contract details: {format_date(picked['expiry'])} "
                        f"strike {picked['strike']:g} {picked['option_type']}",
                        f"📌 合约详情:{format_date(picked['expiry'])} "
                        f"行权价 {picked['strike']:g} {picked['option_type']}",
                    )
                )
                render_option_detail(
                    conn,
                    with_our_iv(db.get_contract_history(
                        conn, selected_ticker, picked["expiry"], picked["strike"],
                        picked["option_type"], days=history_days, source=page_source,
                    )),
                    selected_ticker, get_tracked(),
                    picked["expiry"], picked["strike"], picked["option_type"],
                    key_prefix="screener",
                )
                is_new_scroll_target = st.session_state.get(scroll_target_key) != selected_contract
                st.session_state[scroll_target_key] = selected_contract
                if is_new_scroll_target:
                    components.html(
                        """<script>
                        const heading = Array.from(
                            window.parent.document.querySelectorAll('h3')
                        ).find(el => el.textContent.includes('Contract details') || el.textContent.includes('合约详情'));
                        if (heading) { heading.scrollIntoView({behavior: 'smooth', block: 'start'}); }
                        </script>""",
                        height=0,
                    )

if active_view == "Unusual Activity":
    st.subheader(i18n.t("Unusual Activity (latest snapshot)", "异动(最新快照)"))
    # Only the latest snapshot is loaded; the baseline comes from the table the
    # worker rebuilds once a day. The rules that decide what is unusual stay in
    # metrics — only the aggregation moved.
    flagged = metrics.unusual_activity(
        latest_df, stats=db.get_volume_stats(conn, selected_ticker, source=page_source)
    )
    st.caption(i18n.t(f"Contracts found: {len(flagged)}", f"命中合约:{len(flagged)} 个"))
    st.dataframe(flagged, use_container_width=True)
    with st.expander(i18n.t("ℹ️ How to read this", "ℹ️ 怎么读这张图")):
        st.write(
            i18n.t(
                "A contract is flagged when today's volume statistically significantly exceeds "
                f"(z-score above {config.UNUSUAL_Z_THRESHOLD}) this specific contract's own "
                "history — not a blanket multiplier applied to every strike. Contracts with "
                f"volume below {config.UNUSUAL_MIN_VOLUME} never make the list regardless of the "
                "statistics — that's noise from illiquid far strikes. Contracts without enough "
                "history use the crude check volume > 2×open interest instead. Rows are sorted "
                "by z-score — the most unusual on top. This is not a sign of insider activity — "
                "a large trade can be part of a hedge for an entirely different position.\n\n"
                "The data source's volume is cumulative since the start of the trading session, "
                "so the history for the mean/standard deviation is built from one (the latest) "
                "snapshot per calendar day — multiple same-day collections are never mixed "
                "together when computing the statistics.",
                "某合约今日成交量在统计上显著超过(z-score 高于 "
                f"{config.UNUSUAL_Z_THRESHOLD})它自己的历史时才会被标出 — "
                "不是对所有行权价一刀切的倍数。成交量低于 "
                f"{config.UNUSUAL_MIN_VOLUME} 的合约无论统计多显著都不会上榜 — "
                "那是远端流动性差的噪声。历史不足的合约退化为粗判:成交量 > 2×持仓量。"
                "按 z-score 排序,最异常的在最上面。这并不代表内幕交易 — "
                "一笔大单完全可能是对另一个仓位做对冲的一部分。\n\n"
                "数据源的成交量是自当日开盘起累计的,因此均值/标准差的历史每天只取一条"
                "(当天最后一次)快照 — 同日多次采集永不混入同一统计。",
            )
        )

if active_view == "OI Delta":
    st.subheader(i18n.t("Open Interest change (latest day vs. previous)",
                        "持仓量变化(最新交易日 vs 前一交易日)"))
    # Two calendar days is the entire input to a day-over-day difference, and
    # the days are named by the run log before any snapshot row is read.
    delta = metrics.oi_delta(
        db.get_snapshots_at(
            conn, selected_ticker,
            db.get_collection_moments(
                conn, selected_ticker, days=history_days, per_day=True, source=page_source
            )[:2],
            source=page_source,
        )
    )
    if delta.empty:
        st.info(i18n.t("Data from at least two distinct calendar days is needed to compute the OI change.",
                       "计算持仓变化至少需要两个不同日历日的数据。"))
    else:
        st.dataframe(delta, use_container_width=True)
        with st.expander(i18n.t("ℹ️ How to read this", "ℹ️ 怎么读这张图")):
            st.write(
                i18n.t(
                    "`open_interest` — the current value as of the latest day, for scale: a "
                    "1000-contract increase is a lot at OI=2000 and barely noticeable at "
                    "OI=200000. `oi_delta` — the absolute signed change: positive — new "
                    "positions opened at this strike over the last calendar day (OI grew), "
                    "negative — some positions closed. `oi_delta_pct` — the same change as a "
                    "percentage of the previous value (empty if the previous OI was 0 — growth "
                    "\"from zero\" has no percentage form). Rows are sorted by the magnitude of "
                    "the absolute delta, not by percentage or sign — the largest OI moves in "
                    "either direction are on top. It is precisely the large moves (regardless "
                    "of sign) that most often form new dealer hedging-flow levels (see the Max "
                    "Pain/GEX tab).\n\n"
                    "The data source's open interest barely updates intraday, so the comparison "
                    "is between the latest snapshots of two distinct calendar days rather than "
                    "the two most recent collections — if you collected data several times "
                    "today, only the freshest collection of the day is used for the comparison.",
                    "`open_interest` — 最新一天的当前值,用来衡量尺度:OI=2000 时增加 1000 张"
                    "是大事,OI=200000 时几乎无感。`oi_delta` — 带符号的绝对变化:"
                    "正数 — 最近一个日历日在该行权价开了新仓(持仓增长),负数 — 有仓位平掉。"
                    "`oi_delta_pct` — 相对前值的百分比变化(前值为 0 时留空 — "
                    "“从零增长”没有百分比形式)。按绝对变化量排序,而非百分比或符号 — "
                    "两个方向上最大的持仓变动都在最上面。恰恰是这些大变动(不分方向)"
                    "最常形成新的做市商对冲位(见“最大痛点 / GEX”视图)。\n\n"
                    "数据源的持仓量盘中几乎不更新,因此比较的是两个不同日历日各自的最新快照,"
                    "而不是最近两次采集 — 如果今天采集了多次,只有当天最新的那次参与比较。",
                )
            )
