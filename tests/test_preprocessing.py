"""Checks for train-only preprocessing and robust inference."""
import unittest
import numpy as np
import pandas as pd
from src.preprocessing import (
    PropertyFeatureBuilder, GroupMedianAreaImputer, QuantileCapper,
    make_preprocessor, split_sale_data, FORBIDDEN_INPUTS,
)


def row(**overrides):
    value = dict(
        id=1, type="Konut", sub_type="Daire", start_date="1/1/19",
        end_date=None, listing_type=1, tom=10, building_age="6-10 arası",
        total_floor_count="5", floor_no="2", room_count="2+1",
        size=100.0, address="City/District/Area", furnished=np.nan,
        heating_type="Klima", price=250000.0, price_currency="TRY",
    )
    value.update(overrides)
    return value


class PreprocessingTests(unittest.TestCase):
    def test_feature_builder_parses_and_excludes_leakage(self):
        source = pd.DataFrame([row(size=1.0), row(id=2, size=120.0)])
        result = PropertyFeatureBuilder().transform(source)
        self.assertTrue(pd.isna(result.iloc[0]["size_m2"]))
        self.assertEqual(result.iloc[0]["size_missing"], 1.0)
        self.assertEqual(result.iloc[0]["building_age_years"], 8.0)
        self.assertEqual(result.iloc[0]["bedrooms"], 2.0)
        self.assertEqual(result.iloc[0]["living_rooms"], 1.0)
        self.assertFalse(set(FORBIDDEN_INPUTS) & set(result.columns))

    def test_group_median_fallback_uses_training_only(self):
        train = pd.DataFrame({
            "district": ["A/X", "A/X", "B/Y", "B/Y"],
            "city": ["A", "A", "B", "B"],
            "sub_type": ["Daire", "Daire", "Villa", "Villa"],
            "size_m2": [80.0, 100.0, 200.0, 220.0],
        })
        imputer = GroupMedianAreaImputer(min_group_size=2).fit(train)
        incoming = pd.DataFrame({
            "district": ["A/X", "New/Z", "New/Z"],
            "city": ["A", "New", "New"],
            "sub_type": ["Daire", "Villa", "Unknown"],
            "size_m2": [np.nan, np.nan, np.nan],
        })
        values = imputer.transform(incoming)["size_m2"].tolist()
        self.assertEqual(values, [90.0, 210.0, 150.0])
        self.assertEqual(imputer.global_median_, 150.0)

    def test_cap_bounds_are_not_refit_on_validation(self):
        train = pd.DataFrame({"size_m2": [20., 30., 40., 100.]})
        capper = QuantileCapper(columns=["size_m2"], lower=.25, upper=.75).fit(train)
        bound = capper.bounds_["size_m2"]
        value = capper.transform(pd.DataFrame({"size_m2": [99999.]})).iloc[0, 0]
        self.assertEqual(value, bound[1])
        self.assertEqual(capper.bounds_["size_m2"], bound)

    def test_group_split_disjoint_and_repeatable(self):
        data = pd.DataFrame([
            row(id=i + 1, address="City/D%d/Area" % (i // 3),
                size=float(80 + i // 3), price=float(100000 + i * 100))
            for i in range(90)
        ])
        result1 = split_sale_data(data, random_state=42)
        result2 = split_sale_data(data, random_state=42)
        self.assertEqual(
            [X.id.tolist() for X, _ in result1],
            [X.id.tolist() for X, _ in result2],
        )
        keys = ["address", "sub_type", "room_count", "size"]
        groups = [set(pd.util.hash_pandas_object(X[keys], index=False)) for X, _ in result1]
        self.assertTrue(groups[0].isdisjoint(groups[1]))
        self.assertTrue(groups[0].isdisjoint(groups[2]))
        self.assertTrue(groups[1].isdisjoint(groups[2]))

    def test_unseen_category_and_missing_area(self):
        train = pd.DataFrame([
            row(id=i + 1, address="C/D%d/A" % i, size=float(80 + i * 5))
            for i in range(8)
        ])
        prep = make_preprocessor(scale_numeric=True, min_frequency=2)
        fitted = prep.fit_transform(train)
        new = pd.DataFrame([row(id=99, address="Other/New/A",
                                sub_type="Other", size=np.nan)])
        transformed = prep.transform(new)
        self.assertEqual(transformed.shape[1], fitted.shape[1])
        values = transformed.data if hasattr(transformed, "data") else transformed
        self.assertTrue(np.isfinite(values).all())


if __name__ == "__main__":
    unittest.main()
