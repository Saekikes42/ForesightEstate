"""Leakage-safe feature preparation for the Zingat housing project.

All learned statistics live in sklearn transformers and are fitted on train only.
Original CSV and raw feature values are never modified in place.
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit

RANDOM_STATE = 42
NUMERIC_FEATURES = [
    "size_m2", "bedrooms", "living_rooms", "building_age_years",
    "building_floors", "floor_number", "floor_ratio",
    "size_missing", "age_missing", "is_new_building", "is_top_floor",
]
CATEGORICAL_FEATURES = ["city", "district", "sub_type", "heating_type", "floor_kind"]
CLIP_FEATURES = ["size_m2"]
FORBIDDEN_INPUTS = ["id", "price", "end_date", "tom", "price_currency", "listing_type", "furnished", "original_currency", "exchange_rate", "rate_date"]


def select_sale_try(data: pd.DataFrame) -> pd.DataFrame:
    """Return sale-like TRY listings with a valid positive target."""
    mask = (data["listing_type"].eq(1)
            & data["price_currency"].eq("TRY")
            & data["price"].gt(0))
    return data.loc[mask].copy()


def split_sale_data(data: pd.DataFrame, random_state: int = RANDOM_STATE):
    """Group-aware ~70/15/15 split; similar listing signatures stay together.

    Groups use only input fields, never the target price. This reduces the
    chance that the same property template appears in both train and test.
    """
    selected = select_sale_try(data)
    X = selected.drop(columns=["price"])
    y = selected["price"].astype(float)
    group_cols = ["address", "sub_type", "room_count", "size"]
    groups = pd.util.hash_pandas_object(X[group_cols], index=False)
    first = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=random_state)
    dev_pos, test_pos = next(first.split(X, y, groups=groups))
    X_dev, y_dev, groups_dev = X.iloc[dev_pos], y.iloc[dev_pos], groups.iloc[dev_pos]
    second = GroupShuffleSplit(
        n_splits=1, test_size=0.15 / 0.85, random_state=random_state + 1
    )
    train_pos, val_pos = next(second.split(X_dev, y_dev, groups=groups_dev))
    return (
        (X_dev.iloc[train_pos], y_dev.iloc[train_pos]),
        (X_dev.iloc[val_pos], y_dev.iloc[val_pos]),
        (X.iloc[test_pos], y.iloc[test_pos]),
    )


def _age_years(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text == "40 ve üzeri":
        return 40.0  # lower bound, not an exact age
    match = re.fullmatch(r"(\d+)-(\d+) arası", text)
    if match:
        return (float(match.group(1)) + float(match.group(2))) / 2
    try:
        return float(text)
    except ValueError:
        return np.nan


class PropertyFeatureBuilder(BaseEstimator, TransformerMixin):
    """Stateless parsing of fields known at listing time."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        required = ["address", "sub_type", "heating_type", "room_count",
                    "building_age", "total_floor_count", "floor_no", "size"]
        absent = sorted(set(required) - set(X.columns))
        if absent:
            raise ValueError("Missing input columns: " + ", ".join(absent))
        address = X["address"].fillna("").astype(str).str.split("/", n=2, expand=True)
        city = address[0].replace("", np.nan)
        district_name = address[1].replace("", np.nan) if 1 in address.columns else pd.Series(np.nan, index=X.index)
        district = (city + "/" + district_name).where(district_name.notna())

        rooms = X["room_count"].astype("string").str.extract(r"^(\d+)\+(\d+)$")
        bedrooms = pd.to_numeric(rooms[0], errors="coerce")
        living_rooms = pd.to_numeric(rooms[1], errors="coerce")

        age = X["building_age"].map(_age_years).astype(float)
        floors_raw = X["total_floor_count"].astype("string")
        floors = pd.to_numeric(floors_raw, errors="coerce")
        floors = floors.fillna(floors_raw.map({"10-20 arası": 15.0, "20 ve üzeri": 20.0}))
        floor_raw = X["floor_no"].astype("string")
        floor_exact = pd.to_numeric(floor_raw, errors="coerce")
        floor_number = floor_exact.copy()
        floor_number = floor_number.fillna(floor_raw.map({
            "Yüksek Giriş": 1, "Giriş Katı": 0, "Zemin Kat": 0,
            "Bahçe katı": 0, "20 ve üzeri": 20
        }))
        basement = pd.to_numeric(floor_raw.str.extract(r"^Kot (\d+)$")[0], errors="coerce")
        floor_number = floor_number.fillna(-basement)
        kind = pd.Series("other", index=X.index, dtype=object)
        kind.loc[floor_exact.notna()] = "numbered"
        kind.loc[floor_raw.str.contains(r"^Kot \d+$", na=False)] = "basement"
        kind.loc[floor_raw.isin(["Giriş Katı", "Zemin Kat", "Yüksek Giriş"])] = "entrance"
        kind.loc[floor_raw.eq("Bahçe katı")] = "garden"
        kind.loc[floor_raw.isin(["Çatı Katı", "Teras Kat", "En Üst Kat"])] = "top"
        kind.loc[floor_raw.eq("Asma Kat")] = "mezzanine"
        kind.loc[floor_raw.isin(["Müstakil", "Komple"])] = "whole_building"
        kind.loc[floor_raw.isna()] = np.nan

        area = pd.to_numeric(X["size"], errors="coerce").where(lambda s: s.ge(15))
        result = pd.DataFrame(index=X.index)
        result["size_m2"] = area
        result["size_missing"] = area.isna().astype(float)
        result["bedrooms"] = bedrooms
        result["living_rooms"] = living_rooms
        result["building_age_years"] = age
        result["age_missing"] = age.isna().astype(float)
        result["is_new_building"] = age.eq(0).astype(float)
        result["building_floors"] = floors.astype(float)
        result["floor_number"] = floor_number.astype(float)
        result["floor_ratio"] = floor_number.div(floors.where(floors.gt(0)))
        floors_exact = pd.to_numeric(floors_raw, errors="coerce")
        result["is_top_floor"] = (floor_exact.notna() & floors_exact.notna()
                                  & floor_exact.eq(floors_exact)).astype(float)
        result["city"] = city.astype(object)
        result["district"] = district.astype(object)
        result["sub_type"] = X["sub_type"].astype(object)
        result["heating_type"] = X["heating_type"].astype(object)
        result["floor_kind"] = kind.astype(object)
        return result[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


class GroupMedianAreaImputer(BaseEstimator, TransformerMixin):
    """Train-only area medians: district/type, city/type, type, global."""

    def __init__(self, min_group_size=20):
        self.min_group_size = min_group_size

    def fit(self, X, y=None):
        data = X[["district", "city", "sub_type", "size_m2"]].dropna(subset=["size_m2"])
        self.global_median_ = float(data["size_m2"].median())
        if not np.isfinite(self.global_median_):
            raise ValueError("No valid area values in training data")
        def table(keys, minimum):
            grouped = data.groupby(keys, dropna=True)["size_m2"].agg(["count", "median"]).reset_index()
            grouped = grouped.loc[grouped["count"].ge(minimum), keys + ["median"]]
            return grouped.rename(columns={"median": "_median"})
        self.district_type_ = table(["district", "sub_type"], self.min_group_size)
        self.city_type_ = table(["city", "sub_type"], self.min_group_size)
        self.type_ = table(["sub_type"], 1)
        return self

    def transform(self, X):
        result = X.copy()
        missing = result["size_m2"].isna()
        if not missing.any():
            return result
        subset = result.loc[missing, ["district", "city", "sub_type"]]
        values = np.full(len(subset), np.nan, dtype=float)
        for keys, lookup in [
            (["district", "sub_type"], self.district_type_),
            (["city", "sub_type"], self.city_type_),
            (["sub_type"], self.type_),
        ]:
            candidate = subset[keys].merge(lookup, on=keys, how="left", sort=False)["_median"].to_numpy()
            values = np.where(np.isnan(values), candidate, values)
        result.loc[missing, "size_m2"] = np.where(np.isnan(values), self.global_median_, values)
        return result


class QuantileCapper(BaseEstimator, TransformerMixin):
    """Clip numeric input features using bounds learned only from train."""

    def __init__(self, columns=None, lower=0.001, upper=0.999):
        self.columns = columns
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        if not 0 <= self.lower < self.upper <= 1:
            raise ValueError("Quantiles must satisfy 0 <= lower < upper <= 1")
        columns = CLIP_FEATURES if self.columns is None else self.columns
        self.bounds_ = {}
        for col in columns:
            values = pd.to_numeric(X[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(values):
                self.bounds_[col] = tuple(map(float, values.quantile([self.lower, self.upper])))
        return self

    def transform(self, X):
        result = X.copy()
        for col, (low, high) in self.bounds_.items():
            result[col] = pd.to_numeric(result[col], errors="coerce").clip(low, high)
        return result


def make_preprocessor(scale_numeric=True, min_frequency=100):
    """Ready-to-fit preprocessing; all fitted values are learned in fit(X_train)."""
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numerical = Pipeline(numeric_steps)
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                min_frequency=min_frequency, sparse_output=True)),
    ])
    encoded = ColumnTransformer([
        ("numeric", numerical, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ], sparse_threshold=0.3)
    return Pipeline([
        ("features", PropertyFeatureBuilder()),
        ("area_imputer", GroupMedianAreaImputer(min_group_size=20)),
        ("capper", QuantileCapper()),
        ("encode", encoded),
    ])
