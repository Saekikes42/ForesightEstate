"""Historical TCMB exchange rates and reproducible TRY target conversion."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://www.tcmb.gov.tr/kurlar"
CURRENCIES = ("EUR", "USD", "GBP")


def fetch_daily_rate(day, timeout=20):
    """Fetch official TCMB indicative ForexBuying rates for one publication day."""
    day = pd.Timestamp(day).normalize()
    url = f"{BASE_URL}/{day:%Y%m}/{day:%d%m%Y}.xml"
    response = requests.get(url, timeout=timeout)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    root = ET.fromstring(response.content)
    xml_day = pd.to_datetime(root.attrib["Tarih"], format="%d.%m.%Y")
    if xml_day != day:
        raise ValueError(f"TCMB date mismatch: {url}")
    result = []
    for node in root.findall("Currency"):
        currency = node.attrib.get("Kod")
        if currency not in CURRENCIES:
            continue
        unit = float(node.findtext("Unit"))
        value = float(node.findtext("ForexBuying"))
        if not (unit > 0 and np.isfinite(value) and value > 0):
            raise ValueError(f"Invalid rate for {currency} on {day.date()}")
        result.append({"rate_date": day, "currency": currency, "try_per_unit": value / unit})
    if len(result) != len(CURRENCIES):
        raise ValueError(f"Missing currency in {url}")
    return result


def download_rates(start_date, end_date, max_workers=10):
    """Download all dates in range; unpublished days are absent from result."""
    days = pd.date_range(pd.Timestamp(start_date).normalize(),
                         pd.Timestamp(end_date).normalize(), freq="D")
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetch_daily_rate, day): day for day in days}
        for future in as_completed(futures):
            rows = future.result()
            if rows:
                results.extend(rows)
    rates = pd.DataFrame(results).sort_values(["currency", "rate_date"]).reset_index(drop=True)
    if rates.empty or rates.duplicated(["currency", "rate_date"]).any():
        raise ValueError("Missing or duplicate TCMB rates")
    return rates


def convert_sale_prices(data, rates, max_lag_days=7):
    """Normalize sale-like listings to TRY, using latest rate on/before start_date.

    Rates are external fixed data, not estimated from train/validation/test.
    Returns a copy; unmatched foreign listings raise instead of silently dropping.
    """
    selected = data.loc[data["listing_type"].eq(1) & data["price"].gt(0)].copy()
    supported = selected["price_currency"].isin(["TRY", *CURRENCIES])
    if not supported.all():
        raise ValueError(f"Unsupported currencies: {selected.loc[~supported, 'price_currency'].unique()}")
    selected["original_currency"] = selected["price_currency"]
    selected["listing_date"] = pd.to_datetime(
        selected["start_date"], format="%m/%d/%y", errors="coerce"
    )
    if selected["listing_date"].isna().any():
        raise ValueError("Unparseable listing dates")
    selected["rate_date"] = pd.NaT
    selected["exchange_rate"] = 1.0
    foreign = selected["original_currency"].ne("TRY")
    rates = rates.copy()
    rates["rate_date"] = pd.to_datetime(rates["rate_date"]).dt.normalize()
    rates = rates.sort_values("rate_date")
    for currency in CURRENCIES:
        mask = selected["original_currency"].eq(currency)
        if not mask.any():
            continue
        left = selected.loc[mask, ["listing_date"]].sort_values("listing_date").reset_index().rename(columns={"index": "row_index"})
        right = rates.loc[rates["currency"].eq(currency), ["rate_date", "try_per_unit"]]
        right = right.sort_values("rate_date")
        if right.empty:
            raise ValueError(f"No TCMB rates for {currency}")
        joined = pd.merge_asof(left, right, left_on="listing_date",
                               right_on="rate_date", direction="backward",
                               tolerance=pd.Timedelta(days=max_lag_days))
        if joined["try_per_unit"].isna().any():
            raise ValueError(f"No prior TCMB rate within {max_lag_days} days for {currency}")
        selected.loc[joined["row_index"].to_numpy(), "rate_date"] = joined["rate_date"].to_numpy()
        selected.loc[joined["row_index"].to_numpy(), "exchange_rate"] = joined["try_per_unit"].to_numpy()
    if selected.loc[foreign, "exchange_rate"].isna().any():
        raise ValueError("Unconverted foreign listing")
    selected["price"] = selected["price"].astype(float) * selected["exchange_rate"]
    selected["price_currency"] = "TRY"
    return selected.drop(columns=["listing_date"])
