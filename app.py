# =========================
# streamlit_lake_app.py (lite & fast)
# =========================
import io
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# statsmodels
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

# sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# opzionali
def _xgb_available():
    try:
        import xgboost  # noqa
        return True
    except Exception:
        return False

# scipy
from scipy.stats import jarque_bera, skew, kurtosis, shapiro

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Lake Bilancino — Fast App", layout="wide")
st.title("Lake Bilancino — Time Series (Ottimizzata)")

# ---------------------------
# Helpers
# ---------------------------
def safe_rmse(y_true, y_pred):
    try:
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        return np.sqrt(mean_squared_error(y_true, y_pred))

def as_daily(series):
    """Daily freq + time interpolation (cheap & stable)."""
    return series.asfreq("D").interpolate("time")

def downsample_for_plot(series: pd.Series, max_points=2000):
    s = series.dropna()
    if len(s) <= max_points:
        return s
    step = max(1, len(s) // max_points)
    return s.iloc[::step]

def missing_heatmap(df, title, max_rows=2000):
    view = df
    if len(df) > max_rows:
        step = max(1, len(df) // max_rows)
        view = df.iloc[::step]
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(view.isna(), cbar=False, ax=ax)
    ax.set_title(title)
    st.pyplot(fig)

# ---------------------------
# Caching: dati e feature ingegnerizzate
# ---------------------------
@st.cache_data
def load_data_both(path_or_buf):
    raw = pd.read_csv(path_or_buf)
    raw["Date"] = pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce")
    raw = raw.dropna(subset=["Date"]).sort_values("Date").set_index("Date")

    # copia PRIMA imputazione (per inspection)
    raw_for_missing = raw.copy()

    # imputazione "leggera" per lavorare comodi
    lake = raw.copy()
    rf_cols = ["Rainfall_S_Piero","Rainfall_Mangona","Rainfall_S_Agata","Rainfall_Cavallina","Rainfall_Le_Croci"]
    for c in ["Lake_Level","Flow_Rate","Temperature_Le_Croci"] + rf_cols:
        if c in lake.columns:
            lake[c] = pd.to_numeric(lake[c], errors="coerce")

    if "Flow_Rate" in lake:
        lake["Flow_Rate"] = lake["Flow_Rate"].interpolate("time").ffill().bfill()
    ex_rf = [c for c in rf_cols if c in lake.columns]
    if ex_rf:
        lake[ex_rf] = lake[ex_rf].interpolate("time").ffill().bfill()
    if "Temperature_Le_Croci" in lake:
        lake["Temperature_Le_Croci"] = lake["Temperature_Le_Croci"].interpolate("time").ffill().bfill()

    # usa float32 per alleggerire memoria
    for c in lake.columns:
        if pd.api.types.is_float_dtype(lake[c]):
            lake[c] = lake[c].astype("float32")

    return raw_for_missing, lake

@st.cache_data
def make_weekly_features(lake: pd.DataFrame):
    rf_cols = [c for c in ["Rainfall_S_Piero","Rainfall_Mangona","Rainfall_S_Agata","Rainfall_Cavallina","Rainfall_Le_Croci"] if c in lake.columns]
    lake_w = lake.resample("W-MON").agg({
        "Lake_Level": "mean",
        "Flow_Rate": "sum",
        **({c: "sum" for c in rf_cols} if rf_cols else {}),
        "Temperature_Le_Croci": "mean",
    }).dropna()

    rain_total_w = lake_w[rf_cols].sum(axis=1) if rf_cols else pd.Series(0, index=lake_w.index)
    X_w = pd.DataFrame(index=lake_w.index)
    X_w["flow_l1"]     = lake_w["Flow_Rate"].shift(1)
    X_w["rain_sum_4w"] = rain_total_w.rolling(4, min_periods=1).sum().shift(1)
    X_w["temp_l1"]     = lake_w["Temperature_Le_Croci"].shift(1)
    doy = lake_w.index.dayofyear
    X_w["sin_y"] = np.sin(2*np.pi*doy/365.25)
    X_w["cos_y"] = np.cos(2*np.pi*doy/365.25)
    y_w = lake_w["Lake_Level"]
    df = pd.concat([y_w, X_w], axis=1).dropna()
    return df["Lake_Level"].astype("float32"), df.drop(columns=["Lake_Level"]).astype("float32")

# ---------------------------
# Caching: modelli
# ---------------------------
@st.cache_resource
def fit_arima(series: pd.Series, order):
    return ARIMA(series, order=order).fit()

@st.cache_resource
def fit_sarimax(y_tr: pd.Series, X_tr: pd.DataFrame, order=(1,0,0), seasonal_order=(1,1,0,52)):
    return SARIMAX(
        y_tr, exog=X_tr, order=order, seasonal_order=seasonal_order,
        enforce_stationarity=False, enforce_invertibility=False
    ).fit(method="lbfgs", maxiter=300, disp=False)

@st.cache_resource
def fit_xgb_model(X_train, y_train, params: dict):
    from xgboost import XGBRegressor
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model

# ---------------------------
# Data load
# ---------------------------
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Carica Lake_Bilancino.csv (oppure lascia vuoto)", type=["csv"])
default_path = "Lake_Bilancino.csv"

if uploaded is not None:
    raw_df, lake = load_data_both(uploaded)
else:
    try:
        raw_df, lake = load_data_both(default_path)
    except Exception:
        st.error("⚠️ File non trovato. Carica il CSV con la sidebar.")
        st.stop()

rf_cols = [c for c in ["Rainfall_S_Piero","Rainfall_Mangona","Rainfall_S_Agata","Rainfall_Cavallina","Rainfall_Le_Croci"] if c in lake.columns]

# ---------------------------
# Sidebar navigation
# ---------------------------
section = st.sidebar.selectbox(
    "Sezione",
    [
        "Data Inspection",
        "Exploratory Data Analysis",
        "Stationarity Tests",
        "ACF/PACF (Correlations)",
        "ARIMA Models",
        "SARIMAX Model",
        "XGBoost Models" if _xgb_available() else "XGBoost Models (xgboost non installato)",
        "Feature Importance" if _xgb_available() else "Feature Importance (xgboost non installato)",
        "Residual Diagnostics",
        "Forecasting",
        "Model Comparison",
    ]
)

# =========================
# Data Inspection
# =========================
if section == "Data Inspection":
    st.header("Data Inspection")

    # info cleaned
    buf = io.StringIO()
    lake.info(buf=buf)
    st.text(buf.getvalue())

    st.write(f"**Date Range:** {lake.index.min().date()} → {lake.index.max().date()}")
    st.write(f"**Estimated Frequency:** {pd.infer_freq(lake.index)}")

    st.subheader("Head (cleaned)")
    st.dataframe(lake.head())

    # Missing BEFORE imputazione - stampa stile print
    st.subheader("Missing values per feature (prima dell'imputazione)")
    missing_counts = raw_df.isnull().sum()
    st.code("Missing values per feature:\n" + missing_counts.to_string())

    # Heatmap BEFORE (campionata per velocità)
    st.subheader("Heatmap dei missing values (prima dell'imputazione)")
    missing_heatmap(raw_df, "Missing BEFORE")

# =========================
# Exploratory Data Analysis
# =========================
elif section == "Exploratory Data Analysis":
    st.header("Exploratory Data Analysis")

    # series downsampled per grafici più veloci
    st.subheader("Lake Level & Flow Rate")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    ax1.plot(downsample_for_plot(lake["Lake_Level"]))
    ax1.set_title("Lake Level"); ax1.set_ylabel("m")
    ax2.plot(downsample_for_plot(lake["Flow_Rate"]))
    ax2.set_title("Flow Rate"); ax2.set_ylabel("m³/s"); ax2.set_xlabel("Date")
    plt.tight_layout(); st.pyplot(fig)

    # Meteo (6 grafici)
    st.subheader("Distribuzione delle feature meteo (time series)")
    met = [c for c in ["Rainfall_S_Piero","Rainfall_Mangona","Rainfall_S_Agata","Rainfall_Cavallina","Rainfall_Le_Croci","Temperature_Le_Croci"] if c in lake.columns]
    if met:
        fig = plt.figure(figsize=(14, 10))
        rows, cols = 3, 2
        for i, col in enumerate(met, 1):
            ax = fig.add_subplot(rows, cols, i)
            if "Rainfall" in col:
                s = downsample_for_plot(lake[col].fillna(0))
                ax.fill_between(s.index, s.values, alpha=0.5); ax.set_ylabel("mm")
            else:
                ax.plot(downsample_for_plot(lake[col]), color="red"); ax.set_ylabel("°C")
            ax.set_title(f"{col}")
        plt.tight_layout(); st.pyplot(fig)

    # Corr matrix (senza annot per velocità)
    st.subheader("Correlation Matrix (cleaned)")
    corr = lake[[c for c in ["Lake_Level","Flow_Rate","Temperature_Le_Croci"] + rf_cols if c in lake.columns]].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=False, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlation Matrix"); st.pyplot(fig)

    # Heatmap AFTER (campionata)
    st.subheader("Heatmap dei missing values (dopo imputazione)")
    missing_heatmap(lake, "Missing AFTER")

# =========================
# Stationarity Tests
# =========================
elif section == "Stationarity Tests":
    st.header("Stationarity Tests")
    lvl = as_daily(lake["Lake_Level"]); fr = as_daily(lake["Flow_Rate"])

    def adf_block(series, name):
        x = series.dropna().values
        stat, pval, lags, nobs, crit, _ = adfuller(x, autolag="AIC")
        out = [f"ADF — {name}",
               f"  Test statistic: {stat:.4f}",
               f"  p-value: {pval:.4f}"]
        for k, v in crit.items():
            out.append(f"  Critical {k}: {v:.4f}")
        out.append("  ✅ Stationary (reject H0)" if pval < 0.05 else "  ❌ NON-stationary (do not reject H0)")
        return "\n".join(out)

    st.text(adf_block(lvl, "Lake_Level"))
    st.text(adf_block(fr,  "Flow_Rate"))

    # PP test opzionale (niente errore se arch manca)
    st.subheader("Phillips–Perron (PP) Test")
    if st.checkbox("Esegui PP test (più lento)", value=False):
        try:
            from arch.unitroot import PhillipsPerron
            @st.cache_data
            def pp_summary(s):
                return PhillipsPerron(s.dropna()).summary().as_text()
            st.code("PP Test — Lake_Level\n" + pp_summary(lvl))
            st.code("PP Test — Flow_Rate\n"  + pp_summary(fr))
        except Exception:
            st.info("Modulo `arch` non installato. Per abilitarlo: `pip install arch`")
    else:
        st.caption("Suggerimento: esegui il PP test solo se necessario.")

# =========================
# ACF/PACF (Correlations)
# =========================
elif section == "ACF/PACF (Correlations)":
    st.header("ACF & PACF")
    lvl = as_daily(lake["Lake_Level"]).dropna()
    fr  = as_daily(lake["Flow_Rate"]).dropna()

    st.subheader("Lake_Level")
    with st.expander("Mostra ACF/PACF — Lake_Level", expanded=False):
        fig, ax = plt.subplots(figsize=(12,4)); plot_acf(lvl, lags=40, ax=ax); ax.set_title("ACF — Lake_Level"); st.pyplot(fig)
        fig, ax = plt.subplots(figsize=(12,4)); plot_pacf(lvl, lags=40, method="ols", ax=ax); ax.set_title("PACF — Lake_Level"); st.pyplot(fig)

    st.subheader("Flow_Rate")
    with st.expander("Mostra ACF/PACF — Flow_Rate", expanded=False):
        fig, ax = plt.subplots(figsize=(12,4)); plot_acf(fr, lags=40, ax=ax); ax.set_title("ACF — Flow_Rate"); st.pyplot(fig)
        fig, ax = plt.subplots(figsize=(12,4)); plot_pacf(fr, lags=40, method="ols", ax=ax); ax.set_title("PACF — Flow_Rate"); st.pyplot(fig)

# =========================
# ARIMA Models (su click, cached)
# =========================
elif section == "ARIMA Models":
    st.header("ARIMA Models")
    lvl = as_daily(lake["Lake_Level"]); fr = as_daily(lake["Flow_Rate"])
    train_lvl = lvl.loc[:'2017-12-31']; test_lvl = lvl.loc['2018-01-01':]
    train_fr  = fr.loc[:'2017-12-31'];  test_fr  = fr.loc['2018-01-01':]

    if st.button("Esegui fit ARIMA"):
        st.session_state["fit_lvl"] = fit_arima(train_lvl, (1,0,0))
        st.session_state["fit_fr"]  = fit_arima(train_fr,  (1,0,1))

    if "fit_lvl" in st.session_state and "fit_fr" in st.session_state:
        fit_lvl = st.session_state["fit_lvl"]; fit_fr = st.session_state["fit_fr"]

        st.subheader("ARIMA(1,0,0) — Lake_Level")
        fc_lvl = pd.Series(fit_lvl.forecast(steps=len(test_lvl)), index=test_lvl.index)
        st.write(f"MAE: {mean_absolute_error(test_lvl, fc_lvl):.3f} · RMSE: {safe_rmse(test_lvl, fc_lvl):.3f}")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(train_lvl, label="Train"); ax.plot(test_lvl, label="Test"); ax.plot(fc_lvl.index, fc_lvl, label="Forecast")
        ax.legend(); st.pyplot(fig)

        st.subheader("ARIMA(1,0,1) — Flow_Rate")
        fc_fr = pd.Series(fit_fr.forecast(steps=len(test_fr)), index=test_fr.index)
        st.write(f"MAE: {mean_absolute_error(test_fr, fc_fr):.3f} · RMSE: {safe_rmse(test_fr, fc_fr):.3f}")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(train_fr, label="Train"); ax.plot(test_fr, label="Test"); ax.plot(fc_fr.index, fc_fr, label="Forecast")
        ax.legend(); st.pyplot(fig)
    else:
        st.info("Premi 'Esegui fit ARIMA' per addestrare i modelli.")

# =========================
# SARIMAX Model (su click, cached)
# =========================
elif section == "SARIMAX Model":
    st.header("SARIMAX Model (Weekly on Lake_Level)")

    y_w, X_w = make_weekly_features(lake)
    split = pd.Timestamp("2017-12-31")
    is_tr = X_w.index <= split
    y_tr, y_te = y_w[is_tr], y_w[~is_tr]
    X_tr, X_te = X_w[is_tr], X_w[~is_tr]

    if st.button("Esegui fit SARIMAX"):
        st.session_state["sarimax_res"] = fit_sarimax(y_tr, X_tr)

    if "sarimax_res" in st.session_state:
        res = st.session_state["sarimax_res"]
        pred = res.get_prediction(start=y_te.index[0], end=y_te.index[-1], exog=X_te)
        y_hat = pred.predicted_mean; ci = pred.conf_int()
        st.write(f"MAE: {mean_absolute_error(y_te, y_hat):.3f} · RMSE: {safe_rmse(y_te, y_hat):.3f} · R²: {r2_score(y_te, y_hat):.3f}")
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(y_tr, label="Train"); ax.plot(y_te, label="Test"); ax.plot(y_hat.index, y_hat, label="Predicted")
        ax.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], alpha=0.15)
        ax.legend(); st.pyplot(fig)
    else:
        st.info("Premi 'Esegui fit SARIMAX' per addestrare il modello.")

# =========================
# XGBoost Models (su click, cached)
# =========================
elif "XGBoost Models" in section:
    st.header("XGBoost Models")
    if not _xgb_available():
        st.info("Installa `xgboost` per abilitare questa sezione: `pip install xgboost`")
    else:
        features = [c for c in ["Rainfall_S_Piero","Rainfall_Mangona","Rainfall_S_Agata","Rainfall_Cavallina","Rainfall_Le_Croci","Temperature_Le_Croci"] if c in lake.columns]
        X = lake[features].copy()
        y_lvl = lake["Lake_Level"].copy()
        y_fr  = lake["Flow_Rate"].copy()

        X_train, X_test = X.loc[:'2017-12-31'], X.loc['2018-01-01':]
        y_lvl_train, y_lvl_test = y_lvl.loc[:'2017-12-31'], y_lvl.loc['2018-01-01':]
        y_fr_train,  y_fr_test  = y_fr.loc[:'2017-12-31'],  y_fr.loc['2018-01-01':]

        params_lvl = dict(n_estimators=150, learning_rate=0.1, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42)
        params_fr  = dict(n_estimators=250, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42)

        if st.button("Esegui fit XGBoost"):
            st.session_state["xgb_lvl"] = fit_xgb_model(X_train, y_lvl_train, params_lvl)
            st.session_state["xgb_fr"]  = fit_xgb_model(X_train, y_fr_train,  params_fr)
            st.session_state["X_all"]   = X

        if "xgb_lvl" in st.session_state and "xgb_fr" in st.session_state:
            xgb_lvl = st.session_state["xgb_lvl"]; xgb_fr = st.session_state["xgb_fr"]

            st.subheader("XGBoost — Lake_Level")
            y_pred_lvl = xgb_lvl.predict(X_test)
            st.write(f"MAE: {mean_absolute_error(y_lvl_test, y_pred_lvl):.3f} · RMSE: {safe_rmse(y_lvl_test, y_pred_lvl):.3f} · R²: {r2_score(y_lvl_test, y_pred_lvl):.3f}")
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(y_lvl_train, label="Train"); ax.plot(y_lvl_test, label="Test"); ax.plot(y_lvl_test.index, y_pred_lvl, label="XGB Forecast")
            ax.legend(); st.pyplot(fig)

            st.subheader("XGBoost — Flow_Rate")
            y_fr_pred = xgb_fr.predict(X_test)
            st.write(f"MAE: {mean_absolute_error(y_fr_test, y_fr_pred):.3f} · RMSE: {safe_rmse(y_fr_test, y_fr_pred):.3f} · R²: {r2_score(y_fr_test, y_fr_pred):.3f}")
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(y_fr_train, label="Train"); ax.plot(y_fr_test, label="Test"); ax.plot(y_fr_test.index, y_fr_pred, label="XGB Forecast")
            ax.legend(); st.pyplot(fig)
        else:
            st.info("Premi 'Esegui fit XGBoost' per addestrare i modelli.")

# =========================
# Feature Importance (riusa XGB)
# =========================
elif section == "Feature Importance":
    st.header("Feature Importance (XGBoost)")
    if not _xgb_available() or "xgb_lvl" not in st.session_state:
        st.info("Esegui prima la sezione 'XGBoost Models'.")
    else:
        X = st.session_state["X_all"]
        features = X.columns.tolist()
        xgb_lvl = st.session_state["xgb_lvl"]
        xgb_fr  = st.session_state["xgb_fr"]

        st.subheader("Lake_Level — Feature Importance")
        fig, ax = plt.subplots(figsize=(8, 5))
        imp = pd.DataFrame({"Feature": features, "Importance": xgb_lvl.feature_importances_}).sort_values("Importance", ascending=False)
        sns.barplot(data=imp, x="Importance", y="Feature", ax=ax)
        ax.set_title("XGB — Lake_Level"); st.pyplot(fig)

        st.subheader("Flow_Rate — Feature Importance")
        fig, ax = plt.subplots(figsize=(8, 5))
        imp = pd.DataFrame({"Feature": features, "Importance": xgb_fr.feature_importances_}).sort_values("Importance", ascending=False)
        sns.barplot(data=imp, x="Importance", y="Feature", ax=ax)
        ax.set_title("XGB — Flow_Rate"); st.pyplot(fig)

# =========================
# Residual Diagnostics (veloce)
# =========================
elif section == "Residual Diagnostics":
    st.header("Residual Diagnostics (ARIMA & SARIMAX)")

    def residual_diag(fit_result, name, lb_lags=(10,20,30), arch_lags=10, shap_max_n=3000):
        r = pd.Series(fit_result.resid).dropna()
        st.write(f"**{name}**  — Mean≈0: {np.round(r.mean(), 6)}  ·  Std: {np.round(r.std(ddof=1), 6)}")
        lb = acorr_ljungbox(r, lags=list(lb_lags), return_df=True)
        st.write(f"Ljung–Box p-values {list(lb_lags)}: {lb['lb_pvalue'].round(4).tolist()}")
        jb_stat, jb_p = jarque_bera(r); st.write(f"Jarque–Bera: {jb_stat:.2f} | p={jb_p:.4f}")
        r_s = r.sample(min(len(r), shap_max_n), random_state=42) if len(r) > shap_max_n else r
        W, p_shap = shapiro(r_s); st.write(f"Shapiro–Wilk: W={W:.4f} | p={p_shap:.4f}")
        lm_stat, lm_p, f_stat, f_p = het_arch(r, nlags=arch_lags)
        st.write(f"ARCH LM (lags={arch_lags}) — LM: {lm_stat:.2f} | p: {lm_p:.4f} (F: {f_stat:.2f} | p: {f_p:.4f})")
        sk = float(skew(r, bias=False)); ku = float(kurtosis(r, fisher=False, bias=False))
        st.write(f"Skew: {sk:.2f} | Kurtosis: {ku:.2f}")
        with st.expander(f"Grafici residui — {name}", expanded=False):
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            axes[0,0].plot(np.arange(len(r)), r.values); axes[0,0].axhline(0, color="k", lw=0.8); axes[0,0].set_title("Residuals")
            sns.histplot(r, kde=True, ax=axes[0,1]); axes[0,1].set_title("Distribution")
            plot_acf(r, lags=40, ax=axes[1,0]); axes[1,0].set_title("ACF")
            qqplot(r, line="s", ax=axes[1,1]); axes[1,1].set_title("Q–Q Plot")
            plt.tight_layout(); st.pyplot(fig)

    # usa fit cached (o addestra al volo)
    lvl = as_daily(lake["Lake_Level"]); fr = as_daily(lake["Flow_Rate"])
    fit_lvl = st.session_state.get("fit_lvl") or fit_arima(lvl.loc[:'2017-12-31'], (1,0,0))
    fit_fr  = st.session_state.get("fit_fr")  or fit_arima(fr.loc[:'2017-12-31'],  (1,0,1))
    residual_diag(fit_lvl, "ARIMA(1,0,0) — Lake_Level")
    residual_diag(fit_fr,  "ARIMA(1,0,1) — Flow_Rate")

    # SARIMAX residuals (riusa fit cached se presente, altrimenti fit veloce su train)
    y_w, X_w = make_weekly_features(lake)
    split = pd.Timestamp("2017-12-31"); is_tr = X_w.index <= split
    sarimax_res = st.session_state.get("sarimax_res") or fit_sarimax(y_w[is_tr], X_w[is_tr])
    residual_diag(sarimax_res, "SARIMAX (1,0,0)×(1,1,0,52) — Lake_Level")

# =========================
# Forecasting (light)
# =========================
elif section == "Forecasting":
    st.header("Forecasting — ARIMA, SARIMAX, XGBoost")

    # ARIMA daily
    lvl = as_daily(lake["Lake_Level"]); fr = as_daily(lake["Flow_Rate"])
    fit_lvl = st.session_state.get("fit_lvl") or fit_arima(lvl.loc[:'2017-12-31'], (1,0,0))
    fit_fr  = st.session_state.get("fit_fr")  or fit_arima(fr.loc[:'2017-12-31'],  (1,0,1))

    def forecast_arima(fit, last_date, steps, title):
        fc = fit.get_forecast(steps=steps)
        mean = fc.predicted_mean; ci = fc.conf_int()
        idx = pd.date_range(last_date + pd.Timedelta(days=1), periods=steps, freq="D")
        mean.index = idx; ci.index = idx
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(mean.index, mean.values, label=f"Forecast {steps}d")
        ax.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], alpha=0.2, label="CI 95%")
        ax.set_title(title); ax.legend(); return fig

    horizon_d = st.selectbox("Orizzonte ARIMA/XGB (giorni)", [15, 30], index=0)
    st.subheader(f"ARIMA — {horizon_d} giorni")
    st.pyplot(forecast_arima(fit_lvl, lvl.index.max(), horizon_d, "Lake_Level — ARIMA(1,0,0)"))
    st.pyplot(forecast_arima(fit_fr,  fr.index.max(),  horizon_d, "Flow_Rate — ARIMA(1,0,1)"))

    # SARIMAX weekly (riusa fit + future exog semplici)
    y_w, X_w = make_weekly_features(lake)
    split = pd.Timestamp("2017-12-31")
    sar_res = st.session_state.get("sarimax_res")
    if sar_res is None:
        is_tr = X_w.index <= split
        sar_res = fit_sarimax(y_w[is_tr], X_w[is_tr])
    st.subheader("SARIMAX — forecast (settimanale)")
    horizon_w = st.selectbox("Orizzonte SARIMAX (settimane)", [8, 12], index=0)
    last_week = X_w.index.max()
    future_idx = pd.date_range(last_week + pd.offsets.Week(weekday=0), periods=horizon_w, freq="W-MON")
    last_row = X_w.iloc[-1].copy()
    X_future = pd.DataFrame([last_row] * horizon_w, index=future_idx)
    doy_f = future_idx.dayofyear
    X_future["sin_y"] = np.sin(2*np.pi*doy_f/365.25); X_future["cos_y"] = np.cos(2*np.pi*doy_f/365.25)
    sar_fc = sar_res.get_forecast(steps=horizon_w, exog=X_future)
    sar_mean = sar_fc.predicted_mean; sar_ci = sar_fc.conf_int()
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(y_w.tail(52), label="Observed (last year)")
    ax.plot(sar_mean.index, sar_mean.values, label=f"SARIMAX forecast {horizon_w}w")
    ax.fill_between(sar_ci.index, sar_ci.iloc[:,0], sar_ci.iloc[:,1], alpha=0.2)
    ax.set_title("Lake_Level — SARIMAX forecast"); ax.legend(); st.pyplot(fig)

    # XGBoost forecast (se disponibile e già addestrato)
    if _xgb_available() and "xgb_lvl" in st.session_state:
        st.subheader(f"XGBoost — forecast ({horizon_d} giorni)")
        X_all = st.session_state["X_all"].copy()
        xgb_lvl = st.session_state["xgb_lvl"]; xgb_fr = st.session_state["xgb_fr"]

        def recursive_forecast_xgb_naive(model, X_all, horizon_days):
            feat_names = getattr(model, "feature_names_in_", X_all.columns)
            X_all = X_all.reindex(columns=feat_names)
            work_last = X_all.iloc[-1].copy(); last_date = X_all.index[-1]
            preds, idxs = [], []
            for h in range(1, horizon_days + 1):
                new_date = last_date + pd.Timedelta(days=h)
                new_row = work_last.copy()  # naive: exog costanti
                y_hat = model.predict(pd.DataFrame([new_row], index=[new_date]))[0]
                preds.append(y_hat); idxs.append(new_date); work_last = new_row
            return pd.Series(preds, index=pd.DatetimeIndex(idxs))

        fc_lvl = recursive_forecast_xgb_naive(xgb_lvl, X_all, horizon_d)
        fc_fr  = recursive_forecast_xgb_naive(xgb_fr,  X_all, horizon_d)

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(lake["Lake_Level"].tail(365), label="Observed (last year)")
        ax.plot(fc_lvl.index, fc_lvl.values, "o-", label=f"XGB forecast {horizon_d}d")
        ax.set_title("Lake_Level — XGBoost forecast"); ax.legend(); st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(lake["Flow_Rate"].tail(365), label="Observed (last year)")
        ax.plot(fc_fr.index, fc_fr.values, "o-", label=f"XGB forecast {horizon_d}d")
        ax.set_title("Flow_Rate — XGBoost forecast"); ax.legend(); st.pyplot(fig)
    else:
        st.caption("Per il forecast XGBoost: addestra i modelli nella sezione 'XGBoost Models'.")

# =========================
# Model Comparison (demo statica)
# =========================
elif section == "Model Comparison":
    st.header("Model Comparison")
    results = {
        "Target": ["Lake_Level", "Lake_Level", "Lake_Level", "Flow_Rate", "Flow_Rate"],
        "Model":  ["ARIMA(1,0,0)", "SARIMAX",  "XGBRegressor", "ARIMA(1,0,1)", "XGBRegressor"],
        "MAE":    [1.9, 1.054, 1.518, 2.9, 2.52],
        "RMSE":   [2.45, 1.316, 1.77, 4.4, 4.048],
    }
    df = pd.DataFrame(results)
    st.dataframe(df.style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}"}).background_gradient(subset=["MAE","RMSE"], cmap="Blues"))
