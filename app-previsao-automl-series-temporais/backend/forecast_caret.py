"""
Machine Learning Time-Series Forecasting Engine (PyCaret + Caret Recipes Architecture).

Supports:
1. Native PyCaret (pycaret.regression.RegressionExperiment) when installed, plus a unified
   PyCaret-compatible AutoML & Caret Recipes Pipeline engine powered by Scikit-Learn & XGBoost
   that runs cleanly across Python 3.10+.
2. Full catalog of 165 model configurations (base, `_scale`, `_boxcox`, `_tune`) covering
   linear, robust, penalized, spline/GAM, SVM, Gaussian Process, tree/ensemble, XGBoost, KNN, and MLP.
3. Strict featurization ordering (chronological split BEFORE fitting preprocessing/recipes).
4. Advanced recipes: Imputation (KNN, Bagged/Iterative, Median, Mean, Mode, Linear),
   Numeric Transformations (Normalize, Center, Scale, BoxCox, YeoJohnson, Log, Sqrt, NZV, Corr, Splines),
   Feature Engineering (Interactions, Polynomials, Lags, Date Features, Rolling Windows, Ratios),
   Dimensionality Reduction (PCA, ICA, PLS, Linear Combinations).
5. Validation strategies: TimeSlices (`TimeSeriesSplit`) and Traditional CV (`KFold` / `RepeatedKFold`)
   plus Auto-Tune (`RandomizedSearchCV` / `GridSearchCV`).
6. External Backtest Recommendation Dashboard (`gerar_dashboard_recomendacao_adaptado`).
7. Variable Importance Waterfall/Cascade and Numerical Elasticities (`calcular_elasticidades_modelo`).
"""
from __future__ import annotations
import importlib.util
import math
import re
import sys
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import FastICA, PCA
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.feature_selection import RFE, VarianceThreshold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, DotProduct, RBF
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import (
    ARDRegression,
    BayesianRidge,
    ElasticNet,
    HuberRegressor,
    Lars,
    Lasso,
    LassoLars,
    LinearRegression,
    OrthogonalMatchingPursuit,
    Ridge,
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    RepeatedKFold,
    TimeSeriesSplit,
    cross_val_score,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    SplineTransformer,
    StandardScaler,
)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

HAS_PYCARET = importlib.util.find_spec("pycaret") is not None
HAS_XGBOOST = importlib.util.find_spec("xgboost") is not None


# ---------------------------------------------------------------------------
# Custom Recipe Transformers (Strict Featurization Ordering - Fit on Train!)
# ---------------------------------------------------------------------------

class SafeBoxCoxTransformer(BaseEstimator, TransformerMixin):
    """Box-Cox transformer that shifts non-positive columns learned strictly on training data."""

    def __init__(self) -> None:
        self.shifts_: Optional[np.ndarray] = None
        self.pt_: Optional[PowerTransformer] = None

    def fit(self, X: np.ndarray, y: Any = None) -> "SafeBoxCoxTransformer":
        X_arr = np.asarray(X, dtype=float)
        mins = np.nanmin(X_arr, axis=0)
        self.shifts_ = np.where(mins <= 0, np.abs(mins) + 1e-4, 0.0)
        X_pos = np.maximum(X_arr + self.shifts_, 1e-6)
        self.pt_ = PowerTransformer(method="box-cox", standardize=True)
        try:
            self.pt_.fit(X_pos)
        except Exception:
            self.pt_ = PowerTransformer(method="yeo-johnson", standardize=True)
            self.pt_.fit(X_arr)
            self.shifts_ = np.zeros(X_arr.shape[1])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        if self.pt_ is None or self.shifts_ is None:
            return X_arr
        if self.pt_.method == "box-cox":
            X_pos = np.maximum(X_arr + self.shifts_, 1e-6)
            return self.pt_.transform(X_pos)
        return self.pt_.transform(X_arr)


class SafeLogTransformer(BaseEstimator, TransformerMixin):
    """Log transformer that shifts non-positive values based on training minimum."""

    def __init__(self, offset: float = 1.0) -> None:
        self.offset = offset
        self.shifts_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: Any = None) -> "SafeLogTransformer":
        X_arr = np.asarray(X, dtype=float)
        mins = np.nanmin(X_arr, axis=0)
        self.shifts_ = np.where(mins + self.offset <= 0, np.abs(mins) + self.offset + 1e-6, self.offset)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        shifts = self.shifts_ if self.shifts_ is not None else self.offset
        return np.log(np.maximum(X_arr + shifts, 1e-9))


class SafeSqrtTransformer(BaseEstimator, TransformerMixin):
    """Square-root transformer handling negative values safely."""

    def __init__(self) -> None:
        self.shifts_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: Any = None) -> "SafeSqrtTransformer":
        X_arr = np.asarray(X, dtype=float)
        mins = np.nanmin(X_arr, axis=0)
        self.shifts_ = np.where(mins < 0, np.abs(mins), 0.0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        shifts = self.shifts_ if self.shifts_ is not None else 0.0
        return np.sqrt(np.maximum(X_arr + shifts, 0.0))


class HighCorrelationFilter(BaseEstimator, TransformerMixin):
    """Removes highly correlated numeric features above threshold (like recipes::step_corr)."""

    def __init__(self, threshold: float = 0.90) -> None:
        self.threshold = threshold
        self.keep_indices_: Optional[List[int]] = None

    def fit(self, X: np.ndarray, y: Any = None) -> "HighCorrelationFilter":
        X_arr = np.asarray(X, dtype=float)
        n_cols = X_arr.shape[1]
        if n_cols <= 1:
            self.keep_indices_ = list(range(n_cols))
            return self
        corr = np.abs(np.corrcoef(X_arr, rowvar=False))
        corr = np.nan_to_num(corr, nan=0.0)
        drop = set()
        for i in range(n_cols):
            if i in drop:
                continue
            for j in range(i + 1, n_cols):
                if j not in drop and corr[i, j] > self.threshold:
                    drop.add(j)
        keep = [i for i in range(n_cols) if i not in drop]
        self.keep_indices_ = keep if len(keep) > 0 else [0]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        if not self.keep_indices_:
            return X_arr
        return X_arr[:, self.keep_indices_]


class PLSTransformer(BaseEstimator, TransformerMixin):
    """Supervised Partial Least Squares dimensionality reduction (like recipes::step_pls)."""

    def __init__(self, n_components: int = 2) -> None:
        self.n_components = n_components
        self.pls_: Optional[PLSRegression] = None

    def fit(self, X: np.ndarray, y: Any = None) -> "PLSTransformer":
        X_arr = np.asarray(X, dtype=float)
        k = max(1, min(self.n_components, X_arr.shape[1], X_arr.shape[0] - 1))
        self.pls_ = PLSRegression(n_components=k)
        self.pls_.fit(X_arr, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        if self.pls_ is None:
            return X_arr
        return self.pls_.transform(X_arr)


# ---------------------------------------------------------------------------
# Catalog of Models (Caret `lista_modelos_padrao` + PyCaret Catalog)
# ---------------------------------------------------------------------------

BASE_MODEL_DESCRIPTIONS: Dict[str, Tuple[str, str]] = {
    # Linear & Robust Regression
    "lm": ("Linear Regression (OLS / Caret lm | PyCaret lr)", "Regressão Linear"),
    "rlm": ("Robust Linear Regression (Huber / Caret rlm | PyCaret huber)", "Regressão Linear"),
    "glmnet": ("Elastic-Net Regularized GLM (Caret glmnet | PyCaret en)", "Regressão Penalizada"),
    "ridge": ("Ridge Regression L2 (Caret ridge | PyCaret ridge)", "Regressão Penalizada"),
    "lasso": ("Lasso Regression L1 (Caret lasso | PyCaret lasso)", "Regressão Penalizada"),
    "enet": ("Elastic Net (Caret enet | PyCaret en)", "Regressão Penalizada"),
    "bayesglm": ("Bayesian Ridge GLM (Caret bayesglm | PyCaret br)", "Regressão Linear"),
    "ard": ("Automatic Relevance Determination (PyCaret ard)", "Regressão Linear"),
    "lar": ("Least Angle Regression (PyCaret lar)", "Regressão Linear"),
    "llar": ("Lasso Least Angle Regression (PyCaret llar)", "Regressão Penalizada"),
    "omp": ("Orthogonal Matching Pursuit (PyCaret omp)", "Regressão Linear"),
    "leapBackward": ("Backward Stepwise Regression (Caret leapBackward)", "Regressão Linear"),
    "leapForward": ("Forward Stepwise Regression (Caret leapForward)", "Regressão Linear"),
    "leapSeq": ("Sequential Feature Selection Regression (Caret leapSeq)", "Regressão Linear"),
    "lmStepAIC": ("Stepwise AIC Linear Model (Caret lmStepAIC)", "Regressão Linear"),
    "glmStepAIC": ("Stepwise AIC GLM (Caret glmStepAIC)", "Regressão Linear"),
    "glm": ("Generalized Linear Model (Caret glm)", "Regressão Linear"),
    "glm.nb": ("Negative Binomial / Poisson GLM (Caret glm.nb)", "Regressão Linear"),
    "BstLm": ("Boosted Linear Model (Caret BstLm)", "Regressão Linear"),
    "glmboost": ("Gradient Boosted GLM (Caret glmboost)", "Regressão Linear"),
    "plsRglm": ("Partial Least Squares Regression (Caret plsRglm)", "Regressão Penalizada"),
    "foba": ("Ridge-Regularized Forward/Backward (Caret foba)", "Regressão Penalizada"),
    "blasso": ("Bayesian Lasso / ARD (Caret blasso)", "Regressão Penalizada"),
    "blassoAveraged": ("Averaged Bayesian Lasso (Caret blassoAveraged)", "Regressão Penalizada"),
    "bridge": ("Bridge Penalized Regression (Caret bridge)", "Regressão Penalizada"),
    "penalized": ("L1/L2 Penalized Regression (Caret penalized)", "Regressão Penalizada"),
    "gamSpline": ("Generalized Additive Model with Splines (Caret gamSpline)", "Regressão Penalizada"),
    "gamLoess": ("Smooth Polynomial/Spline Regression (Caret gamLoess)", "Regressão Penalizada"),
    # SVM & Gaussian Processes
    "svmLinear": ("Support Vector Regression Linear (Caret svmLinear | PyCaret svm)", "SVM & Kernel"),
    "svmLinear2": ("SVR Linear L2 (Caret svmLinear2)", "SVM & Kernel"),
    "svmLinear3": ("SVR Linear Regularized (Caret svmLinear3)", "SVM & Kernel"),
    "svmPoly": ("SVR Polynomial Kernel (Caret svmPoly)", "SVM & Kernel"),
    "svmRadial": ("SVR Radial Basis Function Kernel (Caret svmRadial)", "SVM & Kernel"),
    "gaussprLinear": ("Gaussian Process Linear Kernel (Caret gaussprLinear)", "SVM & Kernel"),
    "gaussprPoly": ("Gaussian Process Polynomial/RBF Kernel (Caret gaussprPoly)", "SVM & Kernel"),
    # Trees, Ensembles & Boosting
    "rf": ("Random Forest Regressor (Caret rf | PyCaret rf)", "Árvores & Ensembles"),
    "ranger": ("Fast Random Forest (Caret ranger)", "Árvores & Ensembles"),
    "Rborist": ("Arborist Random Forest (Caret Rborist)", "Árvores & Ensembles"),
    "et": ("Extra Trees Regressor (PyCaret et)", "Árvores & Ensembles"),
    "gbm": ("Gradient Boosting Machine (Caret gbm | PyCaret gbr)", "Árvores & Ensembles"),
    "xgboost": ("Extreme Gradient Boosting (PyCaret xgboost)", "Árvores & Ensembles"),
    "AdaBag": ("AdaBoost Ensemble (Caret AdaBag | PyCaret ada)", "Árvores & Ensembles"),
    "treebag": ("Bagged Decision Trees (Caret treebag)", "Árvores & Ensembles"),
    "dt": ("Decision Tree Regressor (PyCaret dt)", "Árvores & Ensembles"),
    "knn": ("K-Nearest Neighbors Regressor (PyCaret knn)", "Vizinhos & Redes Neurais"),
    # Neural Networks
    "brnn": ("Bayesian Regularized Neural Network (Caret brnn | PyCaret mlp)", "Vizinhos & Redes Neurais"),
}


def build_lista_modelos_padrao() -> Dict[str, Dict[str, Any]]:
    """Builds the complete model dictionary matching R's `lista_modelos_padrao` + PyCaret."""
    catalog: Dict[str, Dict[str, Any]] = {}

    # 1. Base models (no preprocess)
    for m_id, (label, group) in BASE_MODEL_DESCRIPTIONS.items():
        catalog[m_id] = {
            "method": m_id,
            "preProcess": [],
            "tuneGrid": None,
            "label": f"{m_id} — {label}",
            "group": f"1. Sem Pré-Processamento ({group})",
        }

    # 2. Scale variants (_scale)
    for m_id, (label, group) in BASE_MODEL_DESCRIPTIONS.items():
        name = f"{m_id}_scale"
        catalog[name] = {
            "method": m_id,
            "preProcess": ["scale"],
            "tuneGrid": None,
            "label": f"{name} — {label} [Scale]",
            "group": f"2. Com Padronização Scale ({group})",
        }

    # 3. BoxCox variants (_boxcox)
    for m_id, (label, group) in BASE_MODEL_DESCRIPTIONS.items():
        name = f"{m_id}_boxcox"
        catalog[name] = {
            "method": m_id,
            "preProcess": ["BoxCox"],
            "tuneGrid": None,
            "label": f"{name} — {label} [Box-Cox]",
            "group": f"3. Com Transformação Box-Cox ({group})",
        }

    # 4. Pre-configured Tuned variants
    tuned_variants = {
        "gaussprPoly_2": ("gaussprPoly", ["scale"], {"kernel": [ConstantKernel(1.0) * (DotProduct(sigma_0=1.0) ** 2)], "alpha": [1e-2, 1e-1]}),
        "gaussprPoly_3": ("gaussprPoly", ["scale"], {"kernel": [ConstantKernel(1.0) * (DotProduct(sigma_0=1.0) ** 3)], "alpha": [1e-2, 1e-1]}),
        "gaussprPoly_4": ("gaussprPoly", ["scale"], {"kernel": [ConstantKernel(1.0) * (DotProduct(sigma_0=1.0) ** 4)], "alpha": [1e-2, 1e-1]}),
        "gaussprPoly_5": ("gaussprPoly", ["scale"], {"kernel": [ConstantKernel(1.0) * (DotProduct(sigma_0=1.0) ** 5)], "alpha": [1e-2, 1e-1]}),
        "svmPoly_2": ("svmPoly", [], {"degree": [2], "C": [0.05, 0.5, 1.0]}),
        "svmPoly_3": ("svmPoly", [], {"degree": [3], "C": [0.05, 0.5, 1.0]}),
        "svmPoly_4": ("svmPoly", [], {"degree": [4], "C": [0.05, 0.5, 1.0]}),
        "svmPoly_5": ("svmPoly", [], {"degree": [5], "C": [0.05, 0.5, 1.0]}),
        "BstLm_scale_tune": ("BstLm", ["scale"], {"n_estimators": [50, 100, 150, 200], "learning_rate": [0.05, 0.1, 0.2]}),
        "lassoscale_tune": ("lasso", ["scale"], {"alpha": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]}),
        "plsRglm_tune": ("plsRglm", [], {"n_components": [1, 2, 3, 5, 7]}),
        "ridge_tune": ("ridge", [], {"alpha": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]}),
        "leapBackward_scale_tune": ("leapBackward", ["scale"], {}),
        "leapForward_scale_tune": ("leapForward", ["scale"], {}),
        "enet_scale_tune": ("enet", ["scale"], {"alpha": [0.01, 0.1, 0.5, 1.0], "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]}),
        "ranger_tune_scale": ("ranger", ["scale"], {"n_estimators": [100, 200], "min_samples_leaf": [2, 5, 10]}),
        "gamSpline_tune_scale": ("gamSpline", ["scale"], {}),
        "svmPoly_scale_tune": ("svmPoly", ["scale"], {"degree": [1, 2, 3], "C": [0.1, 0.5, 1.0]}),
        "glmnet_tune_scale": ("glmnet", ["scale"], {"alpha": [0.01, 0.05, 0.1, 0.5, 1.0], "l1_ratio": [0.1, 0.5, 0.9]}),
        "glmnet_tune_boxcox": ("glmnet", ["BoxCox"], {"alpha": [0.01, 0.05, 0.1, 0.5, 1.0], "l1_ratio": [0.1, 0.5, 0.9]}),
        "lasso_tune_scale": ("lasso", ["scale"], {"alpha": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]}),
        "lasso_tune_boxcox": ("lasso", ["BoxCox"], {"alpha": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]}),
        "ridge_tune_scale": ("ridge", ["scale"], {"alpha": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]}),
        "ridge_tune_boxcox": ("ridge", ["BoxCox"], {"alpha": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]}),
        "enet_tune_scale_zv": ("enet", ["scale", "zv"], {"alpha": [0.01, 0.1, 0.5, 1.0], "l1_ratio": [0.2, 0.5, 0.8]}),
        "enet_tune_boxcoxzv": ("enet", ["BoxCox", "zv"], {"alpha": [0.01, 0.1, 0.5, 1.0], "l1_ratio": [0.2, 0.5, 0.8]}),
        "gamSpline_tune_boxcox": ("gamSpline", ["BoxCox"], {}),
    }
    for t_name, (m_id, preproc, grid) in tuned_variants.items():
        catalog[t_name] = {
            "method": m_id,
            "preProcess": preproc,
            "tuneGrid": grid,
            "label": f"{t_name} — Variante Tunada ({m_id})",
            "group": "4. Variantes Tunadas (Caret & PyCaret)",
        }

    return catalog


LISTA_MODELOS_PADRAO = build_lista_modelos_padrao()


def parse_custom_models_txt(text_content: str) -> Dict[str, Dict[str, Any]]:
    """
    Parses an R-style or simple text custom model definition file (`lista_modelos_custom`)
    or newline/comma-separated list of model names.
    """
    custom: Dict[str, Dict[str, Any]] = {}
    # Match entries like: model_name = list(method = "rf", preProcess = c("scale"), ...)
    pattern = re.compile(
        r"([A-Za-z0-9_.]+)\s*=\s*list\s*\(\s*method\s*=\s*[\"']([A-Za-z0-9_.]+)[\"'](.*?)\)",
        re.DOTALL,
    )
    for match in pattern.finditer(text_content):
        name = match.group(1).strip()
        method = match.group(2).strip()
        rest = match.group(3)
        preproc: List[str] = []
        if "scale" in rest:
            preproc.append("scale")
        if "BoxCox" in rest or "boxcox" in rest:
            preproc.append("BoxCox")
        if "center" in rest:
            preproc.append("center")
        if "zv" in rest:
            preproc.append("zv")
        custom[name] = {
            "method": method,
            "preProcess": preproc,
            "tuneGrid": None,
            "label": f"{name} (Custom: {method})",
            "group": "Customizados (.txt)",
        }

    if not custom:
        # Try simple line-by-line or comma-separated model names
        tokens = [t.strip() for t in re.split(r"[,;\r\n]+", text_content) if t.strip() and not t.strip().startswith("#")]
        for tok in tokens:
            if tok in LISTA_MODELOS_PADRAO:
                custom[tok] = LISTA_MODELOS_PADRAO[tok]
            elif tok in BASE_MODEL_DESCRIPTIONS:
                custom[tok] = {
                    "method": tok,
                    "preProcess": [],
                    "tuneGrid": None,
                    "label": f"{tok} (Custom)",
                    "group": "Customizados (.txt)",
                }
    return custom


def _instantiate_estimator(method: str, n_features: int, random_state: int = 2024) -> Tuple[BaseEstimator, Dict[str, List[Any]]]:
    """
    Maps a Caret / PyCaret method identifier to a Scikit-Learn / XGBoost estimator
    and its default hyperparameter tuning grid.
    """
    m = method.strip()
    max_k = max(1, min(n_features, 8))

    if m in ("lm", "lr", "glm"):
        return LinearRegression(), {"fit_intercept": [True, False]}
    elif m in ("rlm", "huber"):
        return HuberRegressor(max_iter=500), {"epsilon": [1.1, 1.35, 1.5, 1.75], "alpha": [1e-4, 1e-3, 1e-2]}
    elif m in ("ridge", "foba", "bridge"):
        return Ridge(alpha=1.0, random_state=random_state), {"alpha": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]}
    elif m in ("lasso",):
        return Lasso(alpha=0.1, max_iter=3000, random_state=random_state), {"alpha": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]}
    elif m in ("glmnet", "enet", "en", "penalized"):
        return (
            ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=3000, random_state=random_state),
            {"alpha": [0.005, 0.01, 0.05, 0.1, 0.5, 1.0], "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
        )
    elif m in ("bayesglm", "br", "glm.nb"):
        return BayesianRidge(), {"alpha_1": [1e-6, 1e-5, 1e-4], "lambda_1": [1e-6, 1e-5, 1e-4]}
    elif m in ("ard", "blasso", "blassoAveraged"):
        return ARDRegression(), {"alpha_1": [1e-6, 1e-5], "lambda_1": [1e-6, 1e-5]}
    elif m in ("lar",):
        return Lars(n_nonzero_coefs=max_k), {"n_nonzero_coefs": sorted(list({max(1, max_k // 2), max_k}))}
    elif m in ("llar",):
        return LassoLars(alpha=0.05, max_iter=1000), {"alpha": [0.001, 0.01, 0.05, 0.1, 0.5]}
    elif m in ("omp",):
        return OrthogonalMatchingPursuit(n_nonzero_coefs=max_k), {"n_nonzero_coefs": sorted(list({max(1, max_k // 2), max_k}))}
    elif m in ("leapBackward", "lmStepAIC", "glmStepAIC"):
        if n_features > 2:
            return RFE(estimator=LinearRegression(), n_features_to_select=max(1, int(n_features * 0.7))), {
                "n_features_to_select": sorted(list({max(1, n_features // 2), max(1, int(n_features * 0.75)), n_features}))
            }
        return LinearRegression(), {"fit_intercept": [True]}
    elif m in ("leapForward", "leapSeq"):
        if n_features > 2:
            return RFE(estimator=Ridge(alpha=0.5), n_features_to_select=max(1, int(n_features * 0.6))), {
                "n_features_to_select": sorted(list({max(1, n_features // 2), max(1, int(n_features * 0.75)), n_features}))
            }
        return Ridge(alpha=0.5), {"alpha": [0.1, 0.5, 1.0]}
    elif m in ("BstLm", "glmboost"):
        return (
            GradientBoostingRegressor(n_estimators=100, max_depth=2, learning_rate=0.08, random_state=random_state),
            {"n_estimators": [50, 100, 150, 200], "learning_rate": [0.03, 0.08, 0.15], "max_depth": [1, 2, 3]},
        )
    elif m in ("plsRglm",):
        n_comp = max(1, min(3, n_features))
        return PLSRegression(n_components=n_comp), {
            "n_components": sorted(list({1, min(2, n_features), min(4, n_features), min(6, n_features)}))
        }
    elif m in ("gamSpline", "gamLoess"):
        est = Pipeline([
            ("spline", SplineTransformer(n_knots=4, degree=2, include_bias=False)),
            ("ridge", Ridge(alpha=1.0, random_state=random_state)),
        ])
        return est, {"spline__n_knots": [3, 4, 5], "ridge__alpha": [0.1, 1.0, 10.0]}
    elif m in ("svmLinear", "svmLinear2", "svmLinear3", "svm"):
        return SVR(kernel="linear", C=1.0), {"C": [0.05, 0.1, 0.5, 1.0, 5.0, 10.0], "epsilon": [0.01, 0.1, 0.2]}
    elif m in ("svmPoly",):
        return SVR(kernel="poly", degree=2, C=1.0), {"degree": [2, 3], "C": [0.1, 0.5, 1.0, 5.0]}
    elif m in ("svmRadial",):
        return SVR(kernel="rbf", C=1.0, gamma="scale"), {"C": [0.1, 1.0, 5.0, 10.0], "gamma": ["scale", "auto"]}
    elif m in ("gaussprLinear",):
        kernel = ConstantKernel(1.0) * DotProduct()
        return GaussianProcessRegressor(kernel=kernel, alpha=1e-2, normalize_y=True, random_state=random_state), {
            "alpha": [1e-3, 1e-2, 1e-1, 1.0]
        }
    elif m in ("gaussprPoly",):
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        return GaussianProcessRegressor(kernel=kernel, alpha=1e-2, normalize_y=True, random_state=random_state), {
            "alpha": [1e-3, 1e-2, 1e-1, 1.0]
        }
    elif m in ("rf", "ranger", "Rborist"):
        return (
            RandomForestRegressor(n_estimators=150, random_state=random_state, n_jobs=1),
            {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 5, 10, 15],
                "min_samples_leaf": [1, 2, 4, 8],
            },
        )
    elif m in ("et",):
        return (
            ExtraTreesRegressor(n_estimators=150, random_state=random_state, n_jobs=1),
            {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 5, 10, 15],
                "min_samples_leaf": [1, 2, 4],
            },
        )
    elif m in ("gbm", "gbr"):
        return (
            GradientBoostingRegressor(n_estimators=120, learning_rate=0.08, max_depth=3, random_state=random_state),
            {
                "n_estimators": [80, 120, 200],
                "learning_rate": [0.03, 0.08, 0.15],
                "max_depth": [2, 3, 4],
                "subsample": [0.8, 1.0],
            },
        )
    elif m in ("xgboost",):
        if HAS_XGBOOST:
            from xgboost import XGBRegressor
            return (
                XGBRegressor(
                    n_estimators=120,
                    learning_rate=0.08,
                    max_depth=4,
                    random_state=random_state,
                    n_jobs=1,
                    verbosity=0,
                ),
                {
                    "n_estimators": [80, 150, 250],
                    "learning_rate": [0.03, 0.08, 0.15],
                    "max_depth": [3, 4, 6],
                },
            )
        return (
            GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=random_state),
            {"n_estimators": [100, 150, 250], "learning_rate": [0.05, 0.1], "max_depth": [3, 4, 5]},
        )
    elif m in ("AdaBag", "ada"):
        return (
            AdaBoostRegressor(n_estimators=100, learning_rate=0.5, random_state=random_state),
            {"n_estimators": [50, 100, 200], "learning_rate": [0.1, 0.5, 1.0]},
        )
    elif m in ("treebag",):
        return (
            BaggingRegressor(n_estimators=80, random_state=random_state, n_jobs=1),
            {"n_estimators": [50, 80, 150], "max_samples": [0.7, 0.85, 1.0]},
        )
    elif m in ("dt",):
        return (
            DecisionTreeRegressor(random_state=random_state),
            {"max_depth": [3, 5, 8, 12, None], "min_samples_leaf": [1, 2, 5, 10]},
        )
    elif m in ("knn",):
        return KNeighborsRegressor(n_neighbors=5), {"n_neighbors": [3, 5, 7, 10], "weights": ["uniform", "distance"]}
    elif m in ("brnn", "mlp"):
        return (
            MLPRegressor(hidden_layer_sizes=(32, 16), alpha=0.1, max_iter=600, random_state=random_state),
            {"hidden_layer_sizes": [(16,), (32,), (32, 16), (64, 32)], "alpha": [0.01, 0.1, 1.0]},
        )
    else:
        return Ridge(alpha=1.0, random_state=random_state), {"alpha": [0.1, 1.0, 10.0]}


def _apply_dataframe_feature_engineering(
    df_sub: pd.DataFrame,
    explicativas: List[str],
    engenharia_features: Optional[List[str]] = None,
    opcoes_features: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Applies explicit dataframe-level feature engineering (lags, rolling windows, ratios,
    interactions, polynomials, date features) matching R's `recipes` steps.
    """
    out = df_sub.copy()
    current_x = list(explicativas)
    if not engenharia_features:
        return out, current_x

    opts = opcoes_features or {}

    for eng in engenharia_features:
        if eng == "date_features" and "Data" in out.columns:
            dt = pd.to_datetime(out["Data"], errors="coerce")
            for cname, series in [
                ("feat_month", dt.dt.month),
                ("feat_quarter", dt.dt.quarter),
                ("feat_year", dt.dt.year),
            ]:
                if cname not in out.columns:
                    out[cname] = series
                    current_x.append(cname)

        elif eng == "lag":
            lag_vars = opts.get("lag_vars") or current_x[:5]
            lag_periods = opts.get("lag_periods") or [1, 2, 3]
            for var in lag_vars:
                if var in out.columns and pd.api.types.is_numeric_dtype(out[var]):
                    for p in lag_periods:
                        p_int = int(p)
                        cname = f"{var}_lag{p_int}"
                        out[cname] = out[var].shift(p_int).bfill()
                        if cname not in current_x:
                            current_x.append(cname)

        elif eng == "window":
            w_size = int(opts.get("window_size") or 3)
            w_vars = opts.get("window_vars") or current_x[:5]
            for var in w_vars:
                if var in out.columns and pd.api.types.is_numeric_dtype(out[var]):
                    cname = f"{var}_rollmean{w_size}"
                    out[cname] = out[var].rolling(window=w_size, min_periods=1).mean()
                    if cname not in current_x:
                        current_x.append(cname)

        elif eng == "ratio":
            r_vars = opts.get("ratio_vars") or []
            if len(r_vars) >= 2 and r_vars[0] in out.columns and r_vars[1] in out.columns:
                v1, v2 = r_vars[0], r_vars[1]
                denom = out[v2].replace(0, np.nan)
                cname = f"ratio_{v1}_por_{v2}"
                out[cname] = (out[v1] / denom).bfill().ffill().fillna(0.0)
                if cname not in current_x:
                    current_x.append(cname)

        elif eng == "interacoes":
            i_vars = opts.get("vars_to_interact") or current_x[:4]
            valid_ivars = [v for v in i_vars if v in out.columns and pd.api.types.is_numeric_dtype(out[v])]
            for idx_a in range(len(valid_ivars)):
                for idx_b in range(idx_a + 1, len(valid_ivars)):
                    va, vb = valid_ivars[idx_a], valid_ivars[idx_b]
                    cname = f"{va}_x_{vb}"
                    out[cname] = out[va] * out[vb]
                    if cname not in current_x:
                        current_x.append(cname)

        elif eng == "polinomiais":
            deg = int(opts.get("degree") or 2)
            p_vars = opts.get("poly_vars") or current_x[:5]
            for var in p_vars:
                if var in out.columns and pd.api.types.is_numeric_dtype(out[var]):
                    for d in range(2, deg + 1):
                        cname = f"{var}_pow{d}"
                        out[cname] = out[var] ** d
                        if cname not in current_x:
                            current_x.append(cname)

    return out, current_x


def _build_preprocessing_pipeline(
    model_preproc: List[str],
    metodo_impute: Optional[str] = None,
    transformacoes_numericas: Optional[List[str]] = None,
    opcoes_transform: Optional[Dict[str, Any]] = None,
    reducao_dimensao: Optional[str] = None,
    opcoes_reducao: Optional[Dict[str, Any]] = None,
    n_features: int = 1,
) -> List[Tuple[str, BaseEstimator]]:
    """
    Constructs a Scikit-Learn Pipeline of preprocessing steps matching R's `recipes` and `caret::preProcess`.
    Always fitted ONLY on training data to respect strict featurization ordering!
    """
    steps: List[Tuple[str, BaseEstimator]] = []
    opts_t = opcoes_transform or {}
    opts_r = opcoes_reducao or {}

    # 1. Imputation (Always ensure no NaNs reach downstream estimators)
    imp = (metodo_impute or "").strip().lower()
    if imp == "knn":
        k_val = int(opts_t.get("knn_k", 5))
        steps.append(("impute", KNNImputer(n_neighbors=k_val)))
    elif imp in ("bagimpute", "linear"):
        steps.append(("impute", IterativeImputer(random_state=2024, max_iter=10)))
    elif imp == "mean":
        steps.append(("impute", SimpleImputer(strategy="mean")))
    elif imp == "moda":
        steps.append(("impute", SimpleImputer(strategy="most_frequent")))
    else:
        # Default safe median imputer
        steps.append(("impute", SimpleImputer(strategy="median")))

    # 2. Zero variance removal
    steps.append(("zv", VarianceThreshold(threshold=0.0)))

    # Combine model-level preProcess and user-selected recipe transformations (case-insensitive dedup)
    all_transforms: List[str] = []
    seen_transforms = {"zv"}
    for source_list in (model_preproc or [], transformacoes_numericas or []):
        for t in source_list:
            t_norm = t.strip().lower()
            if t_norm and t_norm not in seen_transforms:
                seen_transforms.add(t_norm)
                all_transforms.append(t_norm)

    for t_low in all_transforms:
        if t_low == "nzv":
            steps.append(("nzv", VarianceThreshold(threshold=1e-5)))
        elif t_low == "normalize":
            steps.append(("normalize", MinMaxScaler()))
        elif t_low == "center":
            steps.append(("center", StandardScaler(with_mean=True, with_std=False)))
        elif t_low == "scale":
            steps.append(("scale", StandardScaler(with_mean=True, with_std=True)))
        elif t_low == "boxcox":
            steps.append(("boxcox", SafeBoxCoxTransformer()))
        elif t_low == "yeojohnson":
            steps.append(("yeojohnson", PowerTransformer(method="yeo-johnson", standardize=True)))
        elif t_low == "log":
            steps.append(("log", SafeLogTransformer(offset=float(opts_t.get("log_offset", 1.0)))))
        elif t_low == "sqrt":
            steps.append(("sqrt", SafeSqrtTransformer()))
        elif t_low == "corr":
            thresh = float(opts_t.get("corr_threshold", 0.90))
            steps.append(("corr", HighCorrelationFilter(threshold=thresh)))
        elif t_low == "splines":
            knots = max(2, int(opts_t.get("df_knots", 3)))
            steps.append(("splines", SplineTransformer(n_knots=knots, degree=2, include_bias=False)))

    # 3. Dimensionality Reduction
    red = (reducao_dimensao or "").strip().lower()
    if red and n_features >= 2:
        n_comp = max(1, min(int(opts_r.get("num_comp", 5)), n_features))
        if red == "pca":
            steps.append(("pca", PCA(n_components=n_comp, random_state=2024)))
        elif red == "ica":
            steps.append(("ica", FastICA(n_components=n_comp, random_state=2024, max_iter=300)))
        elif red == "pls":
            steps.append(("pls", PLSTransformer(n_components=n_comp)))
        elif red == "lincomb":
            steps.append(("lincomb", HighCorrelationFilter(threshold=0.98)))

    return steps


def extrair_importancia_modelo(
    pipeline_model: Pipeline,
    feature_names: List[str],
    X_train: np.ndarray,
    y_train: np.ndarray,
    nome_modelo: str,
) -> Optional[Dict[str, Any]]:
    """
    Extracts variable importance or coefficients from any trained Pipeline model,
    normalizes to 100%, and builds the Waterfall/Cascade structure (`criar_grafico_cascata`).
    """
    try:
        est = pipeline_model.named_steps.get("model", pipeline_model)
        # If wrapped in GridSearchCV / RandomizedSearchCV
        if hasattr(est, "best_estimator_"):
            est = est.best_estimator_
            if isinstance(est, Pipeline) and "model" in est.named_steps:
                est = est.named_steps["model"]

        raw_importances: Optional[np.ndarray] = None
        raw_coefs: Optional[np.ndarray] = None
        tipo = "importancia"

        # Check if any dimensionality reduction changed feature count
        if hasattr(est, "feature_importances_"):
            fi = np.asarray(est.feature_importances_, dtype=float)
            if len(fi) == len(feature_names):
                raw_importances = np.abs(fi)
                tipo = "importancia_arvore"
        elif hasattr(est, "coef_"):
            c = np.asarray(est.coef_, dtype=float).flatten()
            if len(c) == len(feature_names):
                raw_coefs = c
                raw_importances = np.abs(c)
                tipo = "coeficientes"

        # Fallback: Permutation importance on the full pipeline (works even with PCA/Splines/etc.!)
        if raw_importances is None or np.sum(raw_importances) == 0:
            perm = permutation_importance(
                pipeline_model,
                X_train,
                y_train,
                n_repeats=5,
                random_state=2024,
                scoring="neg_root_mean_squared_error",
            )
            raw_importances = np.maximum(perm.importances_mean, 0.0)
            if np.sum(raw_importances) == 0:
                raw_importances = np.abs(perm.importances_mean) + 1e-6
            tipo = "importancia_permutacao"

        total = float(np.sum(raw_importances))
        if total <= 0:
            norm_imp = np.full(len(feature_names), 100.0 / max(1, len(feature_names)))
        else:
            norm_imp = (raw_importances / total) * 100.0

        records = []
        for idx, fname in enumerate(feature_names):
            rec: Dict[str, Any] = {
                "Variavel": fname,
                "Importancia_Media": round(float(norm_imp[idx]), 2),
            }
            if raw_coefs is not None and idx < len(raw_coefs):
                rec["Coeficiente"] = round(float(raw_coefs[idx]), 6)
            records.append(rec)

        # Sort descending by Importancia_Media
        records.sort(key=lambda r: r["Importancia_Media"], reverse=True)

        # Compute cumulative waterfall coordinates (Start, End, Index) matching `criar_grafico_cascata`
        cum = 0.0
        for i, rec in enumerate(records):
            val = rec["Importancia_Media"]
            rec["Index"] = i + 1
            rec["Start"] = round(cum, 2)
            cum += val
            rec["End"] = round(cum, 2)

        return {
            "modelo": nome_modelo,
            "tipo": tipo,
            "dados": records,
        }
    except Exception:
        return None


def calcular_elasticidades_modelo(
    pipeline_model: Pipeline,
    X_train_df: pd.DataFrame,
    nome_modelo: str,
) -> Optional[List[Dict[str, Any]]]:
    """
    Port of `calcular_elasticidades` from R/functions_page_forecast_caret.R:
    Perturbs each explanatory numeric variable by +1%, +5%, +10%, +15% and measures
    the median % change in the predicted outcome divided by the % perturbation.
    """
    try:
        X_base = X_train_df.values.astype(float)
        y_pred_orig = np.asarray(pipeline_model.predict(X_base), dtype=float).flatten()
        valid_idx = np.abs(y_pred_orig) > 1e-9
        if not np.any(valid_idx):
            return None

        results: List[Dict[str, Any]] = []
        for col_idx, var_name in enumerate(X_train_df.columns):
            col_vals = X_base[:, col_idx]
            if np.all(np.isnan(col_vals)) or np.nanmean(np.abs(col_vals)) == 0:
                continue

            elast_samples: List[float] = []
            for fator in (1.0, 5.0, 10.0, 15.0):
                X_mod = X_base.copy()
                X_mod[:, col_idx] = X_mod[:, col_idx] * (1.0 + fator / 100.0)
                try:
                    y_pred_mod = np.asarray(pipeline_model.predict(X_mod), dtype=float).flatten()
                    diff_perc = ((y_pred_mod[valid_idx] - y_pred_orig[valid_idx]) / y_pred_orig[valid_idx]) * 100.0
                    elast_val = float(np.nanmean(diff_perc) / fator)
                    if math.isfinite(elast_val):
                        elast_samples.append(elast_val)
                except Exception:
                    continue

            if elast_samples:
                med_elast = round(float(np.median(elast_samples)), 4)
                sinal = "Positivo" if med_elast >= 0 else "Negativo"
                results.append({
                    "Variavel": var_name,
                    "Elasticidade": round(abs(med_elast), 4),
                    "Elasticidade_Com_Sinal": med_elast,
                    "Sinal": sinal,
                })

        results.sort(key=lambda r: r["Elasticidade"], reverse=True)
        return results
    except Exception:
        return None


def gerar_dashboard_recomendacao_adaptado(
    df_projecoes: pd.DataFrame,
    df_reais: pd.DataFrame,
    var_Y: str,
    periodo_maximo: int = 12,
) -> List[Dict[str, Any]]:
    """
    Exact Python port of `gerar_dashboard_recomendacao_adaptado` from R/functions_page_forecast_caret.R.
    Evaluates each model over the most recent `periodo_maximo` realized dates.
    """
    min_meses_requeridos = 3
    if df_projecoes is None or df_reais is None or len(df_projecoes) == 0 or len(df_reais) == 0:
        return []

    reais = df_reais[["Data", "valor_real"]].dropna().copy()
    reais["Data"] = pd.to_datetime(reais["Data"]).dt.strftime("%Y-%m-%d")

    proj = df_projecoes.copy()
    proj["Data"] = pd.to_datetime(proj["Data"]).dt.strftime("%Y-%m-%d")

    modelos = [
        c for c in proj.columns
        if c not in ("Data", var_Y) and c.lower() not in ("realizada", "realizado", "valor_real")
    ]

    resultados: List[Dict[str, Any]] = []
    prefix = f"{var_Y}_"

    for col_modelo in modelos:
        display_name = col_modelo[len(prefix):] if col_modelo.startswith(prefix) else col_modelo
        df_comum = pd.merge(proj[["Data", col_modelo]], reais, on="Data", how="inner")
        df_comum = df_comum.dropna(subset=[col_modelo, "valor_real"])

        rec: Dict[str, Any] = {
            "modelo": display_name,
            "coluna_pred": col_modelo,
            "meses": 0,
            "RMSE": None,
            "MAPE": None,
            "MPE": None,
            "MedAE": None,
            "RMSLE": None,
            "R²": None,
            "Explained_Var": None,
            "desv.máx_perc": None,
            "score": None,
            "status": "SEM DADOS COMUNS",
        }

        if len(df_comum) == 0:
            resultados.append(rec)
            continue

        # Sort descending by date and take the most recent `periodo_maximo` months
        df_comum = df_comum.sort_values("Data", ascending=False).head(int(periodo_maximo))
        meses_analisados = len(df_comum)
        rec["meses"] = meses_analisados

        projetado = df_comum[col_modelo].values.astype(float)
        real = df_comum["valor_real"].values.astype(float)
        erros = real - projetado
        abs_erros = np.abs(erros)

        rec["RMSE"] = round(float(np.sqrt(np.mean(erros ** 2))), 2)
        rec["MedAE"] = round(float(np.median(abs_erros)), 2)

        valid_perc = real != 0
        abs_perc_erros = np.array([], dtype=float)
        if np.any(valid_perc):
            perc_erros = (erros[valid_perc] / real[valid_perc]) * 100.0
            abs_perc_erros = np.abs(perc_erros)
            rec["MAPE"] = round(float(np.mean(abs_perc_erros)), 2)
            rec["MPE"] = round(float(np.mean(perc_erros)), 2)
            rec["desv.máx_perc"] = round(float(np.max(abs_perc_erros)), 2)

        if np.all(projetado >= 0) and np.all(real >= 0):
            msle = float(np.mean((np.log1p(projetado) - np.log1p(real)) ** 2))
            rec["RMSLE"] = round(math.sqrt(msle), 4)

        if meses_analisados >= 2:
            var_real = float(np.var(real, ddof=1))
            if var_real > 1e-10:
                ss_res = float(np.sum(erros ** 2))
                ss_tot = float(np.sum((real - np.mean(real)) ** 2))
                if ss_tot > 1e-10:
                    r_sq = 1.0 - (ss_res / ss_tot)
                    rec["R²"] = round(max(0.0, r_sq), 3)
                var_err = float(np.var(erros, ddof=1))
                expl_var = 1.0 - (var_err / var_real)
                rec["Explained_Var"] = round(max(0.0, expl_var), 3)

        if rec["MAPE"] is not None and rec["desv.máx_perc"] is not None:
            score_comp = rec["MAPE"] + (rec["desv.máx_perc"] * 0.4)
            rec["score"] = round(score_comp, 2)
            desv_acima_5 = int(np.sum(abs_perc_erros > 5.0)) if len(abs_perc_erros) > 0 else 0

            if meses_analisados < min_meses_requeridos:
                rec["status"] = "DADOS INSUF."
            elif score_comp <= 1.5 and desv_acima_5 == 0:
                rec["status"] = "Ótimo"
            elif score_comp <= 2.5 and desv_acima_5 <= 1:
                rec["status"] = "Recomendado"
            elif score_comp <= 3.5 and desv_acima_5 <= 1:
                rec["status"] = "Aceitável"
            elif score_comp <= 5.5:
                rec["status"] = "Avaliar"
            else:
                rec["status"] = "Reprojetar"
        else:
            rec["status"] = "DADOS INSUF." if meses_analisados < min_meses_requeridos else "Erro Cálculo Score"

        resultados.append(rec)

    resultados.sort(key=lambda r: (r["score"] is None, r["score"] if r["score"] is not None else float("inf")))
    return resultados


def treinar_avaliar_modelos_caret_recipes(
    df: pd.DataFrame,
    date_col: str,
    var_Y: str,
    explicativas: List[str],
    modelos_selecionados: List[str],
    data_inicio: str,
    data_teste: str,
    n_test: int = 12,
    metodo_impute: Optional[str] = None,
    transformacoes_numericas: Optional[List[str]] = None,
    opcoes_transform: Optional[Dict[str, Any]] = None,
    engenharia_features: Optional[List[str]] = None,
    opcoes_features: Optional[Dict[str, Any]] = None,
    reducao_dimensao: Optional[str] = None,
    opcoes_reducao: Optional[Dict[str, Any]] = None,
    auto_tune: bool = False,
    tune_length: int = 3,
    validation_method: str = "timeslice",
    cv_folds: int = 5,
    cv_repeats: Optional[int] = None,
    periodo_analise: int = 12,
    custom_models: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Main training, cross-validation, prediction, variable importance, elasticity,
    and backtest recommendation engine.
    """
    if date_col not in df.columns:
        raise ValueError(f"Coluna de data '{date_col}' não encontrada.")
    if var_Y not in df.columns:
        raise ValueError(f"Variável dependente '{var_Y}' não encontrada.")
    if not pd.api.types.is_numeric_dtype(df[var_Y]):
        raise ValueError(f"Variável dependente '{var_Y}' deve ser numérica.")

    explicativas_clean = [x for x in explicativas if x in df.columns and x != var_Y and x != date_col]
    # Keep numeric predictors (or one-hot encode categorical if any)
    work_df = df.copy()
    work_df["Data"] = pd.to_datetime(work_df[date_col], errors="coerce")
    work_df = work_df.dropna(subset=["Data"]).sort_values("Data").reset_index(drop=True)

    # Convert any categorical explanatory variable to dummy/numeric
    final_explicativas: List[str] = []
    for col in explicativas_clean:
        if pd.api.types.is_numeric_dtype(work_df[col]):
            final_explicativas.append(col)
        else:
            work_df[col] = pd.Categorical(work_df[col]).codes.astype(float)
            final_explicativas.append(col)

    # If no explanatory variables were selected, create a numeric time trend index
    if len(final_explicativas) == 0:
        work_df["_trend_auto"] = np.arange(1, len(work_df) + 1, dtype=float)
        final_explicativas = ["_trend_auto"]

    # Apply feature engineering (lags, windows, ratios, interactions, polynomials, date_features)
    work_df, final_explicativas = _apply_dataframe_feature_engineering(
        work_df,
        final_explicativas,
        engenharia_features=engenharia_features,
        opcoes_features=opcoes_features,
    )

    dt_ini = pd.Timestamp(data_inicio)
    dt_fim = pd.Timestamp(data_teste)
    if dt_ini >= dt_fim:
        raise ValueError("A data inicial do treino deve ser anterior à data final.")

    # Strict chronological split: training window is [dt_ini, dt_fim] where Y is not NA
    mask_train = (work_df["Data"] >= dt_ini) & (work_df["Data"] <= dt_fim) & work_df[var_Y].notna()
    df_train = work_df.loc[mask_train, ["Data", var_Y] + final_explicativas].copy()

    # If no explicit imputation was requested, drop rows where Y is NA (predictors are imputed via pipeline fitted on train)
    if len(df_train) < 10:
        raise ValueError(f"Dados insuficientes ({len(df_train)} registros) no período selecionado. Mínimo requerido: 10.")

    # Remove constant columns in df_train (matching `nearZeroVar` in R)
    non_const_x = [
        c for c in final_explicativas
        if df_train[c].notna().sum() > 0 and df_train[c].nunique(dropna=True) > 1
    ]
    if len(non_const_x) == 0:
        df_train["_trend_auto"] = np.arange(1, len(df_train) + 1, dtype=float)
        work_df["_trend_auto"] = np.arange(1, len(work_df) + 1, dtype=float)
        non_const_x = ["_trend_auto"]
    final_explicativas = non_const_x

    X_train_df = df_train[final_explicativas].copy()
    y_train = df_train[var_Y].values.astype(float)
    X_full_df = work_df[final_explicativas].copy()

    # Configure Cross-Validation Splitter (TimeSeriesSplit vs KFold / RepeatedKFold)
    n_samples = len(df_train)
    if validation_method == "cv":
        n_splits = max(2, min(int(cv_folds or 5), n_samples // 2))
        if cv_repeats and int(cv_repeats) > 1:
            cv_splitter: Any = RepeatedKFold(n_splits=n_splits, n_repeats=int(cv_repeats), random_state=2024)
            val_desc = f"Repeated K-Fold CV ({n_splits} folds x {cv_repeats} repetições)"
        else:
            cv_splitter = KFold(n_splits=n_splits, shuffle=True, random_state=2024)
            val_desc = f"K-Fold Cross-Validation ({n_splits} folds)"
    else:
        horizon = max(1, min(int(n_test or 12), n_samples // 4))
        n_splits = max(2, min(5, (n_samples - max(5, int(n_samples * 0.5))) // max(1, horizon)))
        cv_splitter = TimeSeriesSplit(n_splits=n_splits, test_size=horizon)
        val_desc = f"Time Slices (Séries Temporais, horizonte={horizon}, splits={n_splits})"

    catalog = {**LISTA_MODELOS_PADRAO, **(custom_models or {})}

    modelos_treinados: Dict[str, Any] = {}
    importancia_variaveis: Dict[str, Any] = {}
    elasticidades_modelos: Dict[str, Any] = {}
    cv_metrics_list: List[Dict[str, Any]] = []

    df_projecoes = pd.DataFrame({
        "Data": work_df["Data"].dt.strftime("%Y-%m-%d"),
        var_Y: work_df[var_Y].round(4),
    })

    for nome_modelo in modelos_selecionados:
        if nome_modelo not in catalog:
            continue
        m_cfg = catalog[nome_modelo]
        method_id = m_cfg["method"]
        model_preproc = m_cfg.get("preProcess") or []
        custom_grid = m_cfg.get("tuneGrid")

        try:
            prep_steps = _build_preprocessing_pipeline(
                model_preproc=model_preproc,
                metodo_impute=metodo_impute,
                transformacoes_numericas=transformacoes_numericas,
                opcoes_transform=opcoes_transform,
                reducao_dimensao=reducao_dimensao,
                opcoes_reducao=opcoes_reducao,
                n_features=len(final_explicativas),
            )
            base_est, default_grid = _instantiate_estimator(method_id, n_features=len(final_explicativas))
            pipe = Pipeline(prep_steps + [("model", base_est)])

            # Hyperparameter Tuning if auto_tune is enabled or variant has tuneGrid
            active_grid = custom_grid if custom_grid else (default_grid if auto_tune else None)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if active_grid:
                    prefixed_grid = {
                        (k if k.startswith("model__") else f"model__{k}"): v
                        for k, v in active_grid.items()
                    }
                    total_combos = 1
                    for v in prefixed_grid.values():
                        total_combos *= max(1, len(v))
                    n_iter = max(1, min(int(tune_length or 3) * 2, total_combos))
                    if total_combos <= n_iter:
                        search: Any = GridSearchCV(
                            pipe,
                            param_grid=prefixed_grid,
                            cv=cv_splitter,
                            scoring="neg_root_mean_squared_error",
                            n_jobs=1,
                        )
                    else:
                        search = RandomizedSearchCV(
                            pipe,
                            param_distributions=prefixed_grid,
                            n_iter=n_iter,
                            cv=cv_splitter,
                            scoring="neg_root_mean_squared_error",
                            random_state=2024,
                            n_jobs=1,
                        )
                    search.fit(X_train_df.values, y_train)
                    fitted_model = search.best_estimator_
                    cv_rmse = float(-search.best_score_)
                else:
                    scores = cross_val_score(
                        pipe,
                        X_train_df.values,
                        y_train,
                        cv=cv_splitter,
                        scoring="neg_root_mean_squared_error",
                        n_jobs=1,
                    )
                    cv_rmse = float(-np.nanmean(scores))
                    pipe.fit(X_train_df.values, y_train)
                    fitted_model = pipe

                # Predict across entire timeline
                preds = np.asarray(fitted_model.predict(X_full_df.values), dtype=float).flatten()
                train_preds = np.asarray(fitted_model.predict(X_train_df.values), dtype=float).flatten()

            col_pred = f"{var_Y}_{nome_modelo}"
            df_projecoes[col_pred] = np.round(preds, 4)
            modelos_treinados[nome_modelo] = fitted_model

            # In-sample metrics
            train_err = y_train - train_preds
            train_rmse = float(np.sqrt(np.mean(train_err ** 2)))
            train_mae = float(np.mean(np.abs(train_err)))
            ss_tot = float(np.sum((y_train - np.mean(y_train)) ** 2))
            train_r2 = float(1.0 - np.sum(train_err ** 2) / ss_tot) if ss_tot > 1e-10 else 0.0
            nz = y_train != 0
            train_mape = float(np.mean(np.abs(train_err[nz] / y_train[nz])) * 100.0) if np.any(nz) else 0.0

            cv_metrics_list.append({
                "Modelo": nome_modelo,
                "Metodo": method_id,
                "CV_RMSE": round(cv_rmse, 3),
                "Treino_RMSE": round(train_rmse, 3),
                "Treino_MAE": round(train_mae, 3),
                "Treino_MAPE_%": round(train_mape, 2),
                "Treino_R²": round(max(0.0, train_r2), 4),
            })

            # Extract Variable Importance (Waterfall/Cascade) & Elasticities
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                imp_info = extrair_importancia_modelo(
                    fitted_model,
                    feature_names=final_explicativas,
                    X_train=X_train_df.values,
                    y_train=y_train,
                    nome_modelo=nome_modelo,
                )
                if imp_info:
                    importancia_variaveis[nome_modelo] = imp_info

                elast_info = calcular_elasticidades_modelo(
                    fitted_model,
                    X_train_df=X_train_df,
                    nome_modelo=nome_modelo,
                )
                if elast_info:
                    elasticidades_modelos[nome_modelo] = elast_info

        except Exception:
            continue

    if len(modelos_treinados) == 0:
        raise ValueError("Nenhum dos modelos selecionados pôde ser ajustado com os parâmetros fornecidos.")

    # Prepare Realized data for External Backtest Recommendation Dashboard
    df_reais = pd.DataFrame({
        "Data": work_df["Data"].dt.strftime("%Y-%m-%d"),
        "valor_real": work_df[var_Y],
    }).dropna(subset=["valor_real"])

    recomendacao_backtest = gerar_dashboard_recomendacao_adaptado(
        df_projecoes=df_projecoes,
        df_reais=df_reais,
        var_Y=var_Y,
        periodo_maximo=periodo_analise,
    )

    # Mask realized values after data_teste on chart display (matching R's df_plot_data_final)
    dt_fim_str = dt_fim.strftime("%Y-%m-%d")
    chart_series = []
    dates_list = df_projecoes["Data"].tolist()
    real_plot_vals = [
        (None if (d > dt_fim_str or pd.isna(v)) else float(v))
        for d, v in zip(dates_list, df_projecoes[var_Y].tolist())
    ]
    chart_series.append({
        "name": f"Real ({var_Y})",
        "is_real": True,
        "x": dates_list,
        "y": real_plot_vals,
    })

    for nome_modelo in modelos_treinados.keys():
        col_pred = f"{var_Y}_{nome_modelo}"
        vals = [None if pd.isna(v) else float(v) for v in df_projecoes[col_pred].tolist()]
        chart_series.append({
            "name": nome_modelo,
            "is_real": False,
            "x": dates_list,
            "y": vals,
        })

    py_ver = f"Python {sys.version_info.major}.{sys.version_info.minor}"
    config_info = {
        "Engine": "PyCaret + Scikit-Learn Recipes Engine" + (" (PyCaret instalado)" if HAS_PYCARET else f" ({py_ver} Native AutoML)"),
        "Coluna_Data": date_col,
        "Variavel_Dependente_Y": var_Y,
        "Variaveis_Explicativas_X": ", ".join(final_explicativas),
        "Modelos_Treinados": ", ".join(modelos_treinados.keys()),
        "Periodo_Treino_Inicio": dt_ini.strftime("%Y-%m-%d"),
        "Periodo_Treino_Fim": dt_fim.strftime("%Y-%m-%d"),
        "Metodo_Validacao": val_desc,
        "Auto_Tune": "Ativado" if auto_tune else "Desativado",
        "Imputacao": metodo_impute or "Mediana (padrão)",
        "Transformacoes_Numericas": ", ".join(transformacoes_numericas or []) or "Nenhuma",
        "Engenharia_Features": ", ".join(engenharia_features or []) or "Nenhuma",
        "Reducao_Dimensao": reducao_dimensao or "Nenhuma",
        "Data_Execucao": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    df_train_out = df_train.copy()
    df_train_out["Data"] = df_train_out["Data"].dt.strftime("%Y-%m-%d")

    return {
        "var_Y": var_Y,
        "data_inicio": dt_ini.strftime("%Y-%m-%d"),
        "data_teste": dt_fim.strftime("%Y-%m-%d"),
        "modelos_treinados_nomes": list(modelos_treinados.keys()),
        "modelos_objetos": modelos_treinados,
        "df_projecoes": df_projecoes,
        "df_reais": df_reais,
        "df_treino": df_train_out,
        "chart_series": chart_series,
        "cv_metrics": cv_metrics_list,
        "recomendacao_backtest": recomendacao_backtest,
        "importancia_variaveis": importancia_variaveis,
        "elasticidades": elasticidades_modelos,
        "config": config_info,
    }
