# src/preprocess.py
from typing import Dict, List, Any, Optional
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder

class Preprocessor:
    """
    Preprocessor that:
      - cleans and imputes columns (based on rules you specified)
      - creates missing indicators
      - fits a OneHotEncoder on categorical features from training data
      - transforms both train and test into aligned DataFrames ready for modeling

    Usage:
        pre = Preprocessor()
        X_train = pre.fit_transform(train_df, target_col="SalePrice")  # fits stats + encoder
        X_test  = pre.transform(test_df)  # uses fitted stats + encoder (aligned columns)
    """

    def __init__(self, drop_cols: Optional[List[str]] = None):
        # columns to drop (defaults to what you had)
        self.drop_cols = drop_cols or ["PoolQC", "MiscFeature", "Alley", "Fence"]

        # Stats learned from training data
        self.lotfrontage_medians: Optional[pd.Series] = None
        self.electrical_mode: Optional[Any] = None
        self.modes: Dict[str, Any] = {}

        # OneHotEncoder and feature names
        self.categorical_cols: List[str] = []
        self.numeric_cols: List[str] = []
        self.ohe: Optional[OneHotEncoder] = None
        self.ohe_feature_names: List[str] = []

        # columns created by preprocessing (so they exist for train/test)
        self.created_indicator_cols = [
            "MasVnrType_missing",
            "FireplaceQu_missing",
            "LotFrontage_missing",
            "Garage_missing",
            "Bsmt_missing",
        ]

    # ----------------------
    # Fit: compute train stats and fit OHE on cleaned train
    # ----------------------
    def fit(self, df: pd.DataFrame, target_col: Optional[str] = None) -> "Preprocessor":
        """
        Fit preprocessor statistics and fit OneHotEncoder on processed train data.

        Args:
            df: training DataFrame that includes all raw columns (and target if present).
            target_col: optional name of target column to exclude when fitting encoder.
        """
        df = df.copy()

        # 1) compute stats used for imputation
        # LotFrontage median per Neighborhood
        self.lotfrontage_medians = df.groupby("Neighborhood")["LotFrontage"].median()

        # Electrical mode
        self.electrical_mode = df["Electrical"].mode()[0] if "Electrical" in df.columns else None

        # Other modes for test-set-only columns
        cols_mode = ["MSZoning", "Utilities", "Functional",
                     "Exterior1st", "Exterior2nd", "KitchenQual", "SaleType"]
        self.modes = {}
        for col in cols_mode:
            if col in df.columns:
                self.modes[col] = df[col].mode()[0]

        # 2) Transform the training data using the imputation stats we just computed.
        # We call internal _clean (which does not touch encoder) to get a cleaned dataframe.
        cleaned_train = self._clean(df)

        # 3) Determine categorical & numeric columns after cleaning (exclude target if provided)
        if target_col and target_col in cleaned_train.columns:
            cleaned_for_ohe = cleaned_train.drop(columns=[target_col])
        else:
            cleaned_for_ohe = cleaned_train

        # pick categorical columns (object dtype)
        self.categorical_cols = cleaned_for_ohe.select_dtypes(include="object").columns.tolist()
        # numeric columns are the rest
        self.numeric_cols = [c for c in cleaned_for_ohe.columns if c not in self.categorical_cols]

        # 4) Fit OneHotEncoder on categorical columns
        if self.categorical_cols:
            try:
                self.ohe = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
            except TypeError:
                # fallback for older sklearn
                self.ohe = OneHotEncoder(drop="first", handle_unknown="ignore", sparse=False)
            cat_vals = cleaned_for_ohe[self.categorical_cols].astype(str).fillna("None")
            self.ohe.fit(cat_vals)
            # build feature names in order: numeric_cols (in order) + ohe feature names
            ohe_names = self.ohe.get_feature_names_out(self.categorical_cols).tolist()
            self.ohe_feature_names = self.numeric_cols + ohe_names
        else:
            self.ohe = None
            self.ohe_feature_names = self.numeric_cols.copy()

        return self

    # ----------------------
    # Public fit_transform: fit then transform train
    # ----------------------
    def fit_transform(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """
        Fit on train data and return transformed DataFrame (features only, target kept if provided).
        If target_col provided, it's kept as a column in the returned DataFrame.
        """
        self.fit(df, target_col=target_col)
        return self.transform(df, target_col=target_col)

    # ----------------------
    # Transform: apply cleaning + encoder to any (train/test) DataFrame
    # ----------------------
    def transform(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """
        Transform a DataFrame (train or test) using learned statistics and encoder.

        Returns:
            pd.DataFrame: transformed DataFrame with columns aligned to training encoder.
              If target_col is provided and exists in df, it will be appended as the last column.
        """
        df = df.copy()

        # Clean / impute / indicator creation
        cleaned = self._clean(df)

        # If target present, drop it for encoding but keep to reattach later
        target_series = None
        if target_col and target_col in cleaned.columns:
            target_series = cleaned[target_col]
            cleaned = cleaned.drop(columns=[target_col])

        # Ensure all created indicator columns exist (so encoder sees same schema)
        for ind in self.created_indicator_cols:
            if ind not in cleaned.columns:
                cleaned[ind] = 0

        # 1) Numeric (passthrough) part
        numeric_df = cleaned[self.numeric_cols].copy() if self.numeric_cols else pd.DataFrame(index=cleaned.index)

        # 2) Categorical part -> encoded via fitted OHE
        if self.ohe and self.categorical_cols:
            cat_df = cleaned[self.categorical_cols].astype(str).fillna("None")
            cat_arr = self.ohe.transform(cat_df)  # shape (n_samples, n_ohe_features)
            ohe_names = self.ohe.get_feature_names_out(self.categorical_cols).tolist()
            ohe_df = pd.DataFrame(cat_arr, columns=ohe_names, index=cleaned.index)
        else:
            ohe_df = pd.DataFrame(index=cleaned.index)

        # 3) Concat numeric + ohe in the exact order learned during fit
        result = pd.concat([numeric_df, ohe_df], axis=1)

        # 4) Reindex to ensure consistent column order with training
        #    If fit hasn't occurred, ohe_feature_names may be empty; we guard for that.
        if self.ohe_feature_names:
            # If any training columns are missing in result, fill with 0
            result = result.reindex(columns=self.ohe_feature_names, fill_value=0)
        else:
            # no ohe feature names (rare), keep result as is
            result = result

        # 5) Re-attach target column (if requested)
        if target_series is not None:
            result[target_col] = target_series.values

        return result

    # ----------------------
    # Internal cleaning (imputations, indicators) only — used by fit and transform
    # ----------------------
    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform cleaning/imputation steps. Uses learned stats where applicable."""
        df = df.copy()

        # Drop unwanted columns (ignore errors)
        df = df.drop(columns=self.drop_cols, errors="ignore")

        # MasVnrType
        if "MasVnrType" in df.columns:
            df["MasVnrType"] = df["MasVnrType"].fillna("None")
            df["MasVnrType_missing"] = (df["MasVnrType"] == "None").astype(int)

        # FireplaceQu
        if "FireplaceQu" in df.columns:
            df["FireplaceQu"] = df["FireplaceQu"].fillna("None")
            df["FireplaceQu_missing"] = (df["FireplaceQu"] == "None").astype(int)

        # LotFrontage
        if "LotFrontage" in df.columns:
            df["LotFrontage_missing"] = df["LotFrontage"].isna().astype(int)
            if self.lotfrontage_medians is not None:
                # fill with median per neighborhood when NaN
                # use map to avoid apply rowwise for speed
                med_map = self.lotfrontage_medians.to_dict()
                df["LotFrontage"] = df["LotFrontage"].fillna(df["Neighborhood"].map(med_map))

        # Garage categorical columns
        garage_cols_cat = ["GarageFinish", "GarageQual", "GarageType", "GarageCond"]
        for col in garage_cols_cat:
            if col in df.columns:
                df[col] = df[col].fillna("None")
        if "GarageYrBlt" in df.columns:
            df["GarageYrBlt"] = df["GarageYrBlt"].fillna(0)
            df["Garage_missing"] = (df["GarageYrBlt"] == 0).astype(int)

        # Basement categorical columns
        bsmt_cols = ["BsmtFinType1", "BsmtFinType2", "BsmtQual", "BsmtCond", "BsmtExposure"]
        for col in bsmt_cols:
            if col in df.columns:
                df[col] = df[col].fillna("None")
        if "BsmtQual" in df.columns:
            df["Bsmt_missing"] = (df["BsmtQual"] == "None").astype(int)

        # MasVnrArea
        if "MasVnrArea" in df.columns:
            df["MasVnrArea"] = df["MasVnrArea"].fillna(0)

        # Electrical
        if "Electrical" in df.columns and self.electrical_mode is not None:
            df["Electrical"] = df["Electrical"].fillna(self.electrical_mode)

        # Columns that are only missing in test set (use modes from training)
        for col, mode_val in self.modes.items():
            if col in df.columns:
                df[col] = df[col].fillna(mode_val)

        # Numeric columns to fill with zero (test set only)
        num_zero = ["BsmtFullBath", "BsmtHalfBath", "BsmtUnfSF", "BsmtFinSF2",
                    "BsmtFinSF1", "TotalBsmtSF", "GarageCars", "GarageArea"]
        for col in num_zero:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        return df
