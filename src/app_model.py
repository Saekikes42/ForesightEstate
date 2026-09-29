from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"Общая модель": ROOT / "models/app_general.joblib", "Средний сегмент": ROOT / "models/app_middle.joblib"}
METRICS = {"Общая модель": (196033, 0.357), "Средний сегмент": (62140, 0.605)}

class FoldAreaMedianImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        data = X[["neighborhood", "district", "sub_type", "size_m2"]].copy()
        data["size_m2"] = pd.to_numeric(data["size_m2"], errors="coerce")
        valid = data.dropna(subset=["size_m2"])
        nb = valid.groupby(["neighborhood", "sub_type"]).size_m2.agg(["median", "count"])
        ds = valid.groupby(["district", "sub_type"]).size_m2.agg(["median", "count"])
        self.nb_medians_ = nb.loc[nb["count"].ge(20), "median"]
        self.ds_medians_ = ds.loc[ds["count"].ge(20), "median"]
        self.type_medians_ = valid.groupby("sub_type").size_m2.median()
        self.overall_median_ = valid["size_m2"].median()
        self.area_low_, self.area_high_ = valid["size_m2"].quantile([.001, .999])
        return self

    def transform(self, X):
        data = X.copy()
        data["size_m2"] = pd.to_numeric(data["size_m2"], errors="coerce")
        for keys, medians in [
            (["neighborhood", "sub_type"], self.nb_medians_),
            (["district", "sub_type"], self.ds_medians_),
        ]:
            idx = pd.MultiIndex.from_frame(data[keys])
            local = pd.Series(medians.reindex(idx).to_numpy(), index=data.index)
            data["size_m2"] = data["size_m2"].fillna(local)
        data["size_m2"] = (
            data["size_m2"].fillna(data["sub_type"].map(self.type_medians_))
            .fillna(self.overall_median_)
            .clip(self.area_low_, self.area_high_)
        )
        return data

def features(part):
    x = pd.DataFrame(index=part.index)
    a = part.address.fillna("").str.split("/", n=2, expand=True)
    x["city"] = a[0].replace("", np.nan)
    x["district"] = (a[0] + "/" + a[1]).where(a[1].notna())
    x["neighborhood"] = (x["district"] + "/" + a[2]).where(a[2].notna() & a[2].ne(""))
    r = part.room_count.astype("string").str.extract(r"^(\d+)\+(\d+)$")
    x["bedrooms"] = pd.to_numeric(r[0], errors="coerce")
    x["living_rooms"] = pd.to_numeric(r[1], errors="coerce")
    x["rooms_missing"] = x["bedrooms"].isna().astype(int)
    x["size_m2"] = pd.to_numeric(part["size"], errors="coerce")
    valid_area = x.size_m2.ge(15) & x.size_m2.le(100_000)
    valid_area &= ~(x.size_m2.gt(3000) & part.sub_type.isin(["Daire", "Rezidans"]))
    x["size_m2"] = x.size_m2.where(valid_area)
    x["size_missing"] = x.size_m2.isna().astype(int)
    age = part.building_age.astype("string")
    limits = age.str.extract(r"^(\d+)-(\d+) arası$").apply(pd.to_numeric, errors="coerce")
    x["building_age_years"] = pd.to_numeric(age, errors="coerce").fillna(limits.mean(axis=1)).fillna(age.map({"40 ve üzeri": 40}))
    x["age_missing"] = x.building_age_years.isna().astype(int)
    x["is_new_building"] = x.building_age_years.eq(0).fillna(False).astype(int)
    floors = part.total_floor_count.astype("string")
    x["building_floors"] = pd.to_numeric(floors, errors="coerce").fillna(floors.map({"10-20 arası": 15, "20 ve üzeri": 20}))
    floor = part.floor_no.astype("string")
    x["floor_number"] = pd.to_numeric(floor, errors="coerce").fillna(floor.map({"Yüksek Giriş": 1, "Giriş Katı": 0, "Zemin Kat": 0, "Bahçe katı": 0, "20 ve üzeri": 20}))
    basement = pd.to_numeric(floor.str.extract(r"^Kot (\d+)$")[0], errors="coerce")
    x["floor_number"] = x.floor_number.fillna(-basement)
    exact_floor = pd.to_numeric(floor, errors="coerce")
    exact_building = pd.to_numeric(floors, errors="coerce")
    x["floor_conflict"] = (exact_floor.gt(exact_building) & exact_building.notna()).fillna(False).astype(int)
    x.loc[x.floor_conflict.eq(1), "floor_number"] = np.nan
    x["floor_ratio"] = x.floor_number / x.building_floors.where(x.building_floors.gt(0))
    x["is_top_floor"] = (x.floor_number.eq(x.building_floors) & x.building_floors.notna()).fillna(False).astype(int)
    x["sub_type"] = part.sub_type
    x["heating_type"] = part.heating_type
    return x

def load_model(name):
    return joblib.load(MODELS[name])

def predict(bundle, row):
    return float(np.exp(bundle["pipeline"].predict(features(pd.DataFrame([row])))[0]))
