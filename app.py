
import io
import re
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st

# Statsmodels & sklearn
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Optional libs
try:
    from xgboost import XGBRegressor
    XGB_OK = True
except Exception:
    XGB_OK = False

# ---------------------------
# Page config MUST be first
# ---------------------------
st.set_page_config(page_title="Lake Bilancino Time Series Analysis", layout="wide")
st.title("Lake Bilancino — Time Series Analysis")

# ---------------------------
# Load data
# ---------------------------
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload Lake_Bilancino.csv (or keep empty to use default path)", type=["csv"])
default_path = "Lake_Bilancino.csv"

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    # Basic parsing & cleaning
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")
    rf_cols = ["Rainfall_S_Piero", "Rainfall_Mangona", "Rainfall_S_Agata", "Rainfall_Cavallina", "Rainfall_Le_Croci"]
    # Fill/Interpolate
    for c in ["Lake_Level", "Flow_Rate", "Temperature_Le_Croci"] + rf_cols:
        if c in df.columns:
            df[c] = df[c].astype("float64")
    df["Flow_Rate"] = df["Flow_Rate"].interpolate("time").ffill().bfill()
    df[rf_cols] = df[rf_cols].interpolate("time").ffill().bfill()
    df["Temperature_Le_Croci"] = df["Temperature_Le_Croci"].interpolate("time").ffill().bfill()
    return df, rf_cols

if uploaded is not None:
    lake, rf_cols = load_data(uploaded)
else:
    try:
        lake, rf_cols = load_data(default_path)
    except Exception as e:
        st.error("⚠️ Impossibile caricare il dataset. Carica il CSV con la sidebar.")
        st.stop()

# Common helpers
def safe_rmse(y_true, y_pred):
    try:
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        return np.sqrt(mean_squared_error(y_true, y_pred))

def as_daily(series):
    return series.asfreq("D").interpolate("time")

# ---------------------------
# Sidebar navigation
# ---------------------------
section = st.sidebar.selectbox(
    "Sezione",
    [
        "Data Inspection",
        "Exploratory Data Analysis",
        "Stationarity Tests",
        "ARIMA Models",
        "SARIMAX Model",
        "XGBoost Models" if XGB_OK else "XGBoost Models (xgboost non installato)",
        "Residual Diagnostics",
        "Forecasting",
        "Model Comparison",
    ]
)

# ---------------------------
# Sections
# ---------------------------
if section == "Data Inspection":
    st.header("Data Inspection")

    buf = io.StringIO()
    lake.info(buf=buf)
    info_str = buf.getvalue()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Info")
        st.text(info_str)
        st.write(f"**Date Range:** {lake.index.min().date()} → {lake.index.max().date()}")
        st.write(f"**Estimated Frequency:** {pd.infer_freq(lake.index)}")
    with col2:
        st.subheader("Head & Missing")
        st.dataframe(lake.head())
        st.write("Missing values per feature:")
        st.write(lake.isna().sum())

elif section == "Exploratory Data Analysis":
    st.header("Exploratory Data Analysis")

    # Time series plots
    st.subheader("Lake Level & Flow Rate")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    ax1.plot(lake.index, lake["Lake_Level"])
    ax1.set_title("Lake Level")
    ax1.set_ylabel("m")
    ax2.plot(lake.index, lake["Flow_Rate"])
    ax2.set_title("Flow Rate")
    ax2.set_ylabel("m³/s")
    ax2.set_xlabel("Date")
    plt.tight_layout()
    st.pyplot(fig)

    # Correlation
    st.subheader("Correlation Matrix")
    corr = lake.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Matrix")
    st.pyplot(fig)

    # Missing heatmap
    st.subheader("Missing Values Heatmap")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(lake.isna(), cbar=False, cmap="viridis", ax=ax)
    ax.set_title("Missing Values")
    st.pyplot(fig)

elif section == "Stationarity Tests":
    st.header("Stationarity Tests")
    lvl = as_daily(lake["Lake_Level"])
    fr = as_daily(lake["Flow_Rate"])

    def adf_block(series, name):
        out = []
        x = series.dropna().values
        try:
            stat, pval, lags, nobs, crit, _ = adfuller(x, autolag="AIC")
            out.append(f"ADF — {name}")
            out.append(f"  Test statistic: {stat:.4f}")
            out.append(f"  p-value: {pval:.4f}")
            for k, v in crit.items():
                out.append(f"  Critical {k}: {v:.4f}")
            out.append("  ✅ Stationary (reject H0)" if pval < 0.05 else "  ❌ NON-stationary (do not reject H0)")
        except Exception as e:
            out.append(f"ADF failed for {name}: {e}")
        return "\n".join(out)

    st.text(adf_block(lvl, "Lake_Level"))
    st.text(adf_block(fr, "Flow_Rate"))

elif section == "ARIMA Models":
    st.header("ARIMA Models")
    lvl = as_daily(lake["Lake_Level"])
    fr  = as_daily(lake["Flow_Rate"])

    train_lvl = lvl.loc[:'2017-12-31']
    test_lvl  = lvl.loc['2018-01-01':]
    train_fr  = fr.loc[:'2017-12-31']
    test_fr   = fr.loc['2018-01-01':]

    # Lake_Level
    st.subheader("ARIMA(1,0,0) — Lake_Level")
    model_lvl = ARIMA(train_lvl, order=(1,0,0)).fit()
    fc_lvl = pd.Series(model_lvl.forecast(steps=len(test_lvl)), index=test_lvl.index)
    st.write(f"MAE: {mean_absolute_error(test_lvl, fc_lvl):.3f} · RMSE: {safe_rmse(test_lvl, fc_lvl):.3f}")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(train_lvl, label="Train")
    ax.plot(test_lvl, label="Test")
    ax.plot(fc_lvl.index, fc_lvl, label="Forecast", linewidth=2)
    ax.legend(); ax.set_title("ARIMA(1,0,0) — Lake_Level")
    st.pyplot(fig)

    # Flow_Rate
    st.subheader("ARIMA(1,0,1) — Flow_Rate")
    model_fr = ARIMA(train_fr, order=(1,0,1)).fit()
    fc_fr = pd.Series(model_fr.forecast(steps=len(test_fr)), index=test_fr.index)
    st.write(f"MAE: {mean_absolute_error(test_fr, fc_fr):.3f} · RMSE: {safe_rmse(test_fr, fc_fr):.3f}")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(train_fr, label="Train")
    ax.plot(test_fr, label="Test")
    ax.plot(fc_fr.index, fc_fr, label="Forecast", linewidth=2)
    ax.legend(); ax.set_title("ARIMA(1,0,1) — Flow_Rate")
    st.pyplot(fig)

elif section == "SARIMAX Model":
    st.header("SARIMAX Model (Weekly on Lake_Level)")

    rf_cols = [c for c in ["Rainfall_S_Piero","Rainfall_Mangona","Rainfall_S_Agata","Rainfall_Cavallina","Rainfall_Le_Croci"] if c in lake.columns]

    lake_w = lake.resample("W-MON").agg({
        "Lake_Level": "mean",
        "Flow_Rate": "sum",
        **{c: "sum" for c in rf_cols},
        "Temperature_Le_Croci": "mean",
    }).dropna()

    rain_total_w = lake_w[rf_cols].sum(axis=1)
    X_w = pd.DataFrame(index=lake_w.index)
    X_w["flow_l1"] = lake_w["Flow_Rate"].shift(1)
    X_w["rain_sum_4w"] = rain_total_w.rolling(4, min_periods=1).sum().shift(1)
    X_w["temp_l1"] = lake_w["Temperature_Le_Croci"].shift(1)
    doy = lake_w.index.dayofyear
    X_w["sin_y"] = np.sin(2*np.pi*doy/365.25)
    X_w["cos_y"] = np.cos(2*np.pi*doy/365.25)
    y_w = lake_w["Lake_Level"]

    df = pd.concat([y_w, X_w], axis=1).dropna()
    y_w = df["Lake_Level"]; X_w = df.drop(columns=["Lake_Level"])

    split = pd.Timestamp("2017-12-31")
    y_tr, y_te = y_w[X_w.index <= split], y_w[X_w.index > split]
    X_tr, X_te = X_w[X_w.index <= split], X_w[X_w.index > split]

    mod = SARIMAX(y_tr, exog=X_tr, order=(1,0,0), seasonal_order=(1,1,0,52),
                  enforce_stationarity=False, enforce_invertibility=False)
    res = mod.fit(method="lbfgs", maxiter=300, disp=False)
    pred = res.get_prediction(start=y_te.index[0], end=y_te.index[-1], exog=X_te)
    y_hat = pred.predicted_mean
    ci = pred.conf_int()

    mae = mean_absolute_error(y_te, y_hat)
    rmse = safe_rmse(y_te, y_hat)
    r2   = r2_score(y_te, y_hat)
    st.write(f"MAE: {mae:.3f} · RMSE: {rmse:.3f} · R²: {r2:.3f}")

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(y_tr, label="Train", linewidth=1)
    ax.plot(y_te, label="Test", linewidth=1.2)
    ax.plot(y_hat.index, y_hat, label="Predicted", linewidth=1.8)
    ax.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], alpha=0.15)
    ax.set_title("Lake_Level — SARIMAX (1,0,0)x(1,1,0,52)")
    ax.legend()
    st.pyplot(fig)

elif "XGBoost Models" in section:
    st.header(section)
    if not XGB_OK:
        st.info("Installa `xgboost` per abilitare questa sezione: `pip install xgboost`")
    else:
        features = ["Rainfall_S_Piero","Rainfall_Mangona","Rainfall_S_Agata","Rainfall_Cavallina","Rainfall_Le_Croci","Temperature_Le_Croci"]
        features = [c for c in features if c in lake.columns]
        X = lake[features].copy()
        y_lvl = lake["Lake_Level"].copy()
        y_fr  = lake["Flow_Rate"].copy()

        X_train, X_test = X.loc[:'2017-12-31'], X.loc['2018-01-01':]
        y_lvl_train, y_lvl_test = y_lvl.loc[:'2017-12-31'], y_lvl.loc['2018-01-01':]
        y_fr_train, y_fr_test = y_fr.loc[:'2017-12-31'], y_fr.loc['2018-01-01':]

        # Lake_Level
        st.subheader("XGBoost — Lake_Level")
        xgb_lvl = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=4, subsample=0.8,
                               colsample_bytree=0.8, random_state=42)
        xgb_lvl.fit(X_train, y_lvl_train)
        y_pred_lvl = xgb_lvl.predict(X_test)
        st.write(f"MAE: {mean_absolute_error(y_lvl_test, y_pred_lvl):.3f} · RMSE: {safe_rmse(y_lvl_test, y_pred_lvl):.3f} · R²: {r2_score(y_lvl_test, y_pred_lvl):.3f}")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(y_lvl_train, label="Train")
        ax.plot(y_lvl_test, label="Test")
        ax.plot(y_lvl_test.index, y_pred_lvl, label="XGB Forecast")
        ax.legend(); ax.set_title("XGBoost — Lake_Level")
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        imp = pd.DataFrame({"Feature": features, "Importance": xgb_lvl.feature_importances_}).sort_values("Importance", ascending=False)
        sns.barplot(data=imp, x="Importance", y="Feature", ax=ax)
        ax.set_title("Feature Importance — Lake_Level")
        st.pyplot(fig)

        # Flow_Rate
        st.subheader("XGBoost — Flow_Rate")
        xgb_fr = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=5, subsample=0.8,
                              colsample_bytree=0.8, random_state=42)
        xgb_fr.fit(X_train, y_fr_train)
        y_fr_pred = xgb_fr.predict(X_test)
        st.write(f"MAE: {mean_absolute_error(y_fr_test, y_fr_pred):.3f} · RMSE: {safe_rmse(y_fr_test, y_fr_pred):.3f} · R²: {r2_score(y_fr_test, y_fr_pred):.3f}")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(y_fr_train, label="Train")
        ax.plot(y_fr_test, label="Test")
        ax.plot(y_fr_test.index, y_fr_pred, label="XGB Forecast")
        ax.legend(); ax.set_title("XGBoost — Flow_Rate")
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        imp = pd.DataFrame({"Feature": features, "Importance": xgb_fr.feature_importances_}).sort_values("Importance", ascending=False)
        sns.barplot(data=imp, x="Importance", y="Feature", ax=ax)
        ax.set_title("Feature Importance — Flow_Rate")
        st.pyplot(fig)

elif section == "Residual Diagnostics":
    st.header("Residual Diagnostics")
    lvl = as_daily(lake["Lake_Level"])
    fr  = as_daily(lake["Flow_Rate"])
    train_lvl = lvl.loc[:'2017-12-31']
    train_fr  = fr.loc[:'2017-12-31']

    def diag(fit_result, name):
        r = pd.Series(fit_result.resid).dropna()
        st.subheader(name)
        st.write(f"Mean≈0: {r.mean():.6f}  ·  Std: {r.std(ddof=1):.6f}")
        lb = acorr_ljungbox(r, lags=[10,20,30], return_df=True)
        st.write(f"Ljung–Box p-values (10,20,30): {lb['lb_pvalue'].round(4).tolist()}")
        from scipy.stats import jarque_bera, shapiro
        jb_stat, jb_p = jarque_bera(r)
        st.write(f"Jarque–Bera: {jb_stat:.2f}  ·  p={jb_p:.4f}")
        r_s = r.sample(min(len(r), 5000), random_state=42) if len(r) > 5000 else r
        W, p_shap = shapiro(r_s)
        st.write(f"Shapiro–Wilk: W={W:.4f}  ·  p={p_shap:.4f}")

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0,0].plot(np.arange(len(r)), r.values); axes[0,0].axhline(0, color="k", lw=0.8); axes[0,0].set_title("Residuals")
        sns.histplot(r, kde=True, ax=axes[0,1]); axes[0,1].set_title("Distribution")
        plot_acf(r, lags=40, ax=axes[1,0]); axes[1,0].set_title("ACF")
        qqplot(r, line="s", ax=axes[1,1]); axes[1,1].set_title("Q–Q Plot")
        plt.tight_layout()
        st.pyplot(fig)

    model_lvl = ARIMA(train_lvl, order=(1,0,0)).fit()
    model_fr  = ARIMA(train_fr,  order=(1,0,1)).fit()
    diag(model_lvl, "ARIMA(1,0,0) — Lake_Level")
    diag(model_fr,  "ARIMA(1,0,1) — Flow_Rate")

elif section == "Forecasting":
    st.header("Forecasting")
    lvl = as_daily(lake["Lake_Level"])
    fr  = as_daily(lake["Flow_Rate"])
    train_lvl = lvl.loc[:'2017-12-31']
    train_fr  = fr.loc[:'2017-12-31']

    model_lvl = ARIMA(train_lvl, order=(1,0,0)).fit()
    model_fr  = ARIMA(train_fr,  order=(1,0,1)).fit()

    def forecast_arima(fit, last_date, steps, title):
        fc = fit.get_forecast(steps=steps)
        mean = fc.predicted_mean
        ci = fc.conf_int()
        idx = pd.date_range(last_date + pd.Timedelta(days=1), periods=steps, freq="D")
        mean.index = idx; ci.index = idx
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(mean.index, mean.values, label=f"Forecast {steps}d")
        ax.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], alpha=0.2, label="CI 95%")
        ax.set_title(title); ax.legend()
        return fig

    horizon = st.selectbox("Forecast horizon (days)", [15, 30], index=0)
    st.subheader(f"ARIMA — {horizon} Days")
    st.pyplot(forecast_arima(model_lvl, lvl.index.max(), horizon, "Lake_Level — ARIMA(1,0,0)"))
    st.pyplot(forecast_arima(model_fr,  fr.index.max(),  horizon, "Flow_Rate — ARIMA(1,0,1)"))

elif section == "Model Comparison":
    st.header("Model Comparison")
    results = {
        "Target": ["Lake_Level", "Lake_Level", "Lake_Level", "Flow_Rate", "Flow_Rate"],
        "Model": ["ARIMA(1,0,0)", "SARIMAX", "XGBRegressor", "ARIMA(1,0,1)", "XGBRegressor"],
        "MAE": [1.9, 1.054, 1.518, 2.9, 2.52],
        "RMSE": [2.45, 1.316, 1.77, 4.4, 4.048],
    }
    df = pd.DataFrame(results)
    st.dataframe(df.style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}"}).background_gradient(subset=["MAE","RMSE"], cmap="Blues"))
