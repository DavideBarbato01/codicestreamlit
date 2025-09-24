import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.gofplots import qqplot
from xgboost import XGBRegressor
import re
from datetime import timedelta
from scipy.stats import jarque_bera, shapiro

# Streamlit page configuration
st.set_page_config(page_title="Lake Bilancino Time Series Analysis", layout="wide")
st.title("Lake Bilancino Time Series Analysis")

# File uploader for dataset
lake = pd.read_csv("Lake_Bilancino.csv")

# Data Preprocessing
lake["Date"] = pd.to_datetime(lake["Date"], dayfirst=True, errors="coerce")
lake = lake.sort_values("Date").set_index("Date")
lake["Flow_Rate"] = lake["Flow_Rate"].interpolate(method="time").ffill().bfill()
rf_cols = ["Rainfall_S_Piero", "Rainfall_Mangona", "Rainfall_S_Agata", "Rainfall_Cavallina", "Rainfall_Le_Croci"]
lake[rf_cols] = lake[rf_cols].interpolate(method="time").ffill().bfill()
lake["Temperature_Le_Croci"] = lake["Temperature_Le_Croci"].interpolate(method="time").ffill().bfill()

# Sidebar for navigation
st.sidebar.header("Navigation")
section = st.sidebar.selectbox("Select Section", [
    "Data Inspection",
    "Exploratory Data Analysis",
    "Stationarity Tests",
    "ARIMA Models",
    "SARIMAX Model",
    "XGBoost Models",
    "Residual Diagnostics",
    "Forecasting"
])

# Data Inspection
if section == "Data Inspection":
    st.header("Data Inspection")
    st.write("Dataset Information:")
    st.write(lake.info())
    st.write("First 5 rows of the dataset:")
    st.dataframe(lake.head())
    st.write("Missing values per feature:")
    st.write(lake.isnull().sum())
    st.write(f"Date Range: {lake.index.min()} to {lake.index.max()}")
    st.write(f"Estimated Frequency: {pd.infer_freq(lake.index)}")

# Exploratory Data Analysis
elif section == "Exploratory Data Analysis":
    st.header("Exploratory Data Analysis")
    
    # Time series plots
    st.subheader("Time Series Plots")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6))
    ax1.plot(lake.index, lake["Lake_Level"], color="blue")
    ax1.set_title("Lake Bilancino - Lake Level")
    ax1.set_ylabel("Lake_Level (m)")
    ax2.plot(lake.index, lake["Flow_Rate"], color="green")
    ax2.set_title("Lake Bilancino - Flow Rate")
    ax2.set_ylabel("Flow_Rate (m³/s)")
    ax2.set_xlabel("Date")
    plt.tight_layout()
    st.pyplot(fig)

    # Boxplots
    st.subheader("Boxplots")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(y=lake["Lake_Level"], color="skyblue", ax=ax1)
    ax1.set_title("Boxplot - Lake Level")
    sns.boxplot(y=lake["Flow_Rate"], color="lightgreen", ax=ax2)
    ax2.set_title("Boxplot - Flow Rate")
    plt.tight_layout()
    st.pyplot(fig)

    # Time Series of All Variables
    st.subheader("Time Series of All Variables")
    fig, ax = plt.subplots(figsize=(12, 6))
    for column in lake.columns:
        ax.plot(lake.index, lake[column], label=column)
    ax.set_title("Time Series of All Variables")
    ax.legend()
    st.pyplot(fig)

    # Distribution of Variables
    st.subheader("Distribution of Variables")
    num_vars = len(lake.columns)
    fig, axes = plt.subplots(nrows=(num_vars + 1) // 2, ncols=2, figsize=(12, 6 * ((num_vars + 1) // 2)))
    axes = axes.flatten()
    for i, column in enumerate(lake.columns):
        axes[i].hist(lake[column].dropna(), bins=30, color='#1E90FF', edgecolor='black')
        axes[i].set_title(f'Distribution of {column}')
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('Frequency')
    plt.tight_layout()
    st.pyplot(fig)

    # Missing values heatmap
    st.subheader("Missing Values Heatmap")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(lake.isna(), cbar=False, cmap="viridis", ax=ax)
    ax.set_title("Heatmap of Missing Values")
    st.pyplot(fig)

    # Correlation matrix
    st.subheader("Correlation Matrix")
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = lake.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Matrix")
    st.pyplot(fig)

# Stationarity Tests
elif section == "Stationarity Tests":
    st.header("Stationarity Tests")
    lvl = lake["Lake_Level"].asfreq("D").interpolate("time")
    fr = lake["Flow_Rate"].asfreq("D").interpolate("time")

    def adf_report(series, name):
        series = series.dropna()
        stat, pval, lags, nobs, crit, _ = adfuller(series, autolag="AIC")
        result = f"ADF Test — {name}\n"
        result += f"  Test statistic: {stat:.4f}\n"
        result += f"  p-value: {pval:.4f}\n"
        for k, v in crit.items():
            result += f"  Critical {k}: {v:.4f}\n"
        result += "  ✅ Series stationary (reject H0)." if pval < 0.05 else "  ❌ Series NON-stationary (do not reject H0)."
        return result

    st.subheader("ADF Test Results")
    st.text(adf_report(lvl, "Lake_Level"))
    st.text(adf_report(fr, "Flow_Rate"))

# ARIMA Models
elif section == "ARIMA Models":
    st.header("ARIMA Models")
    lvl = lake["Lake_Level"].asfreq("D").interpolate("time")
    fr = lake["Flow_Rate"].asfreq("D").interpolate("time")
    train_lvl = lvl.loc[:'2017-12-31']
    test_lvl = lvl.loc['2018-01-01':]
    train_fr = fr.loc[:'2017-12-31']
    test_fr = fr.loc['2018-01-01':]

    # ARIMA for Lake_Level
    st.subheader("ARIMA(1,0,0) — Lake_Level")
    model_lvl = ARIMA(train_lvl, order=(1,0,0)).fit()
    forecast_lvl = model_lvl.forecast(steps=len(test_lvl))
    forecast_lvl = pd.Series(forecast_lvl, index=test_lvl.index)
    mae_lvl = mean_absolute_error(test_lvl, forecast_lvl)
    import sklearn
    if sklearn.__version__ >= '0.22':
        rmse_lvl = mean_squared_error(test_lvl, forecast_lvl, squared=False)  # RMSE diretto
    else:
        rmse_lvl = np.sqrt(mean_squared_error(test_lvl, forecast_lvl))  # Calcolo manuale per versioni vecchie
    st.write(f"MAE: {mae_lvl:.3f}  RMSE: {rmse_lvl:.3f}")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(train_lvl, label="Train")
    ax.plot(test_lvl, label="Test")
    ax.plot(test_lvl.index, forecast_lvl, label="Forecast", color="red")
    ax.set_title("ARIMA(1,0,0) — Lake_Level")
    ax.legend()
    st.pyplot(fig)

    # ARIMA for Flow_Rate
    st.subheader("ARIMA(1,0,1) — Flow_Rate")
    model_fr = ARIMA(train_fr, order=(1,0,1)).fit()
    forecast_fr = model_fr.forecast(steps=len(test_fr))
    forecast_fr = pd.Series(forecast_fr, index=test_fr.index)
    mae_fr = mean_absolute_error(test_fr, forecast_fr)
    if sklearn.__version__ >= '0.22':
        rmse_fr = mean_squared_error(test_fr, forecast_fr, squared=False)
    else:
        rmse_fr = np.sqrt(mean_squared_error(test_fr, forecast_fr))
    st.write(f"MAE: {mae_fr:.3f}  RMSE: {rmse_fr:.3f}")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(train_fr, label="Train")
    ax.plot(test_fr, label="Test")
    ax.plot(test_fr.index, forecast_fr, label="Forecast", color="red")
    ax.set_title("ARIMA(1,0,1) — Flow_Rate")
    ax.legend()
    st.pyplot(fig)

# SARIMAX Model
elif section == "SARIMAX Model":
    st.header("SARIMAX Model (Weekly, Lake_Level)")
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
    doy_w = pd.Series(lake_w.index.dayofyear, index=lake_w.index)
    X_w["sin_y"] = np.sin(2 * np.pi * doy_w / 365.25)
    X_w["cos_y"] = np.cos(2 * np.pi * doy_w / 365.25)
    y_w = lake_w["Lake_Level"]
    df_w = pd.concat([y_w, X_w], axis=1).dropna()
    y_w = df_w["Lake_Level"]
    X_w = df_w.drop(columns=["Lake_Level"])
    split_date = pd.Timestamp("2017-12-31")
    is_train_w = X_w.index <= split_date
    y_tr, y_te = y_w[is_train_w], y_w[~is_train_w]
    X_tr, X_te = X_w[is_train_w], X_w[~is_train_w]
    order = (1, 0, 0)
    seasonal_order = (1, 1, 0, 52)
    mod = SARIMAX(y_tr, exog=X_tr, order=order, seasonal_order=seasonal_order,
                  enforce_stationarity=False, enforce_invertibility=False)
    res = mod.fit(method="lbfgs", maxiter=300, disp=False)
    pred = res.get_prediction(start=y_te.index[0], end=y_te.index[-1], exog=X_te)
    y_hat = pred.predicted_mean
    ci = pred.conf_int()
    mae = mean_absolute_error(y_te, y_hat)
    if sklearn.__version__ >= '0.22':
        rmse = mean_squared_error(y_te, y_hat, squared=False)
    else:
        rmse = np.sqrt(mean_squared_error(y_te, y_hat))
    r2 = r2_score(y_te, y_hat)
    st.write(f"MAE: {mae:.3f}  RMSE: {rmse:.3f}  R²: {r2:.3f}")
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(y_tr.index, y_tr, label="Train", linewidth=1)
    ax.plot(y_te.index, y_te, label="Test (observed)", linewidth=1.2)
    ax.plot(y_hat.index, y_hat, label="Predicted (SARIMAX s=52)", linewidth=1.8)
    ax.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], alpha=0.15)
    ax.set_title(f"Lake_Level — SARIMAX {order}×{seasonal_order} (weekly)")
    ax.legend()
    st.pyplot(fig)

# XGBoost Models
elif section == "XGBoost Models":
    st.header("XGBoost Models")
    features = rf_cols + ["Temperature_Le_Croci"]
    X = lake[features]
    y_lvl = lake["Lake_Level"]
    y_fr = lake["Flow_Rate"]
    X_train, X_test = X.loc[:'2017-12-31'], X.loc['2018-01-01':]
    y_lvl_train, y_lvl_test = y_lvl.loc[:'2017-12-31'], y_lvl.loc['2018-01-01':]
    y_fr_train, y_fr_test = y_fr.loc[:'2017-12-31'], y_fr.loc['2018-01-01':]

    # XGBoost for Lake_Level
    st.subheader("XGBoost — Lake_Level")
    xgb_lvl = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=4, subsample=0.8,
                           colsample_bytree=0.8, random_state=42)
    xgb_lvl.fit(X_train, y_lvl_train)
    y_pred_lvl = xgb_lvl.predict(X_test)
    mae_xgb = mean_absolute_error(y_lvl_test, y_pred_lvl)
    if sklearn.__version__ >= '0.22':
        rmse_xgb = mean_squared_error(y_lvl_test, y_pred_lvl, squared=False)
    else:
        rmse_xgb = np.sqrt(mean_squared_error(y_lvl_test, y_pred_lvl))
    r2_xgb = r2_score(y_lvl_test, y_pred_lvl)
    st.write(f"MAE: {mae_xgb:.3f}  RMSE: {rmse_xgb:.3f}  R²: {r2_xgb:.3f}")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y_lvl_train, label="Train")
    ax.plot(y_lvl_test, label="Test")
    ax.plot(y_lvl_test.index, y_pred_lvl, label="XGB Forecast", color="red")
    ax.set_title("XGBoost — Lake_Level")
    ax.legend()
    st.pyplot(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    feat_importance = pd.DataFrame({"Feature": features, "Importance": xgb_lvl.feature_importances_})
    sns.barplot(data=feat_importance.sort_values(by="Importance", ascending=False),
                x="Importance", y="Feature", palette="viridis", ax=ax)
    ax.set_title("Feature Importance — Lake_Level")
    st.pyplot(fig)

    # XGBoost for Flow_Rate
    st.subheader("XGBoost — Flow_Rate")
    xgb_fr = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=5, subsample=0.8,
                          colsample_bytree=0.8, random_state=42)
    xgb_fr.fit(X_train, y_fr_train)
    y_fr_pred = xgb_fr.predict(X_test)
    mae_fr = mean_absolute_error(y_fr_test, y_fr_pred)
    if sklearn.__version__ >= '0.22':
        rmse_fr = mean_squared_error(y_fr_test, y_fr_pred, squared=False)
    else:
        rmse_fr = np.sqrt(mean_squared_error(y_fr_test, y_fr_pred))
    st.write(f"MAE: {mae_fr:.3f}  RMSE: {rmse_fr:.3f}")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y_fr_train, label="Train")
    ax.plot(y_fr_test, label="Test")
    ax.plot(y_fr_test.index, y_fr_pred, label="XGB Forecast", color="red")
    ax.set_title("XGBoost — Flow_Rate")
    ax.legend()
    st.pyplot(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    feat_importance = pd.DataFrame({"Feature": features, "Importance": xgb_fr.feature_importances_})
    sns.barplot(data=feat_importance.sort_values(by="Importance", ascending=False),
                x="Importance", y="Feature", palette="viridis", ax=ax)
    ax.set_title("Feature Importance — Flow_Rate")
    st.pyplot(fig)

# Residual Diagnostics
elif section == "Residual Diagnostics":
    st.header("Residual Diagnostics")
    lvl = lake["Lake_Level"].asfreq("D").interpolate("time")
    fr = lake["Flow_Rate"].asfreq("D").interpolate("time")
    train_lvl = lvl.loc[:'2017-12-31']
    train_fr = fr.loc[:'2017-12-31']

    def residual_diagnostics_full(fit_result, name):
        r = pd.Series(fit_result.resid).dropna()
        st.write(f"\n=== Residual Diagnostics — {name} ===")
        st.write(f"Mean (≈0): {np.round(r.mean(), 6)} | Std: {np.round(r.std(ddof=1), 6)}")
        lb = acorr_ljungbox(r, lags=[10, 20, 30], return_df=True)
        st.write(f"Ljung–Box p-values (lags [10, 20, 30]): {lb['lb_pvalue'].round(4).tolist()}")
        jb_stat, jb_p = jarque_bera(r)
        st.write(f"Jarque–Bera (JB): {jb_stat:.2f} | Prob(JB): {jb_p:.4f}")
        r_shap = r.sample(min(len(r), 5000), random_state=42) if len(r) > 5000 else r
        W, p_shap = shapiro(r_shap)
        st.write(f"Shapiro–Wilk: W={W:.4f} | p-value={p_shap:.4f}")
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        sns.lineplot(x=np.arange(len(r)), y=r.values, ax=axes[0, 0])
        axes[0, 0].axhline(0, color="k", lw=0.8)
        axes[0, 0].set_title(f"Residuals — {name}")
        sns.histplot(r, kde=True, ax=axes[0, 1])
        axes[0, 1].set_title("Residual Distribution")
        plot_acf(r, lags=40, ax=axes[1, 0])
        axes[1, 0].set_title("ACF of Residuals")
        qqplot(r, line="s", ax=axes[1, 1])
        axes[1, 1].set_title("Q-Q Plot of Residuals")
        plt.tight_layout()
        st.pyplot(fig)

    model_lvl = ARIMA(train_lvl, order=(1,0,0)).fit()
    model_fr = ARIMA(train_fr, order=(1,0,1)).fit()
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
    doy_w = pd.Series(lake_w.index.dayofyear, index=lake_w.index)
    X_w["sin_y"] = np.sin(2 * np.pi * doy_w / 365.25)
    X_w["cos_y"] = np.cos(2 * np.pi * doy_w / 365.25)
    y_w = lake_w["Lake_Level"]
    df_w = pd.concat([y_w, X_w], axis=1).dropna()
    y_w = df_w["Lake_Level"]
    X_w = df_w.drop(columns=["Lake_Level"])
    is_train_w = X_w.index <= pd.Timestamp("2017-12-31")
    y_tr, X_tr = y_w[is_train_w], X_w[is_train_w]
    mod = SARIMAX(y_tr, exog=X_tr, order=(1, 0, 0), seasonal_order=(1, 1, 0, 52),
                  enforce_stationarity=False, enforce_invertibility=False)
    res = mod.fit(method="lbfgs", maxiter=300, disp=False)

    st.subheader("ARIMA(1,0,0) — Lake_Level")
    residual_diagnostics_full(model_lvl, "ARIMA(1,0,0) — Lake_Level")
    st.subheader("ARIMA(1,0,1) — Flow_Rate")
    residual_diagnostics_full(model_fr, "ARIMA(1,0,1) — Flow_Rate")
    st.subheader("SARIMAX (1,0,0)×(1,1,0,52) — Lake_Level")
    residual_diagnostics_full(res, "SARIMAX (1,0,0)×(1,1,0,52) — Lake_Level")

# Forecasting
elif section == "Forecasting":
    st.header("Forecasting")
    lvl = lake["Lake_Level"].asfreq("D").interpolate("time")
    fr = lake["Flow_Rate"].asfreq("D").interpolate("time")
    train_lvl = lvl.loc[:'2017-12-31']
    train_fr = fr.loc[:'2017-12-31']
    features = rf_cols + ["Temperature_Le_Croci"]
    X = lake[features]
    X_train, X_test = X.loc[:'2017-12-31'], X.loc['2018-01-01':]
    y_lvl_train, y_lvl_test = lvl.loc[:'2017-12-31'], lvl.loc['2018-01-01':]
    y_fr_train, y_fr_test = fr.loc[:'2017-12-31'], fr.loc['2018-01-01':]

    # ARIMA models
    model_lvl = ARIMA(train_lvl, order=(1,0,0)).fit()
    model_fr = ARIMA(train_fr, order=(1,0,1)).fit()

    # XGBoost models
    xgb_lvl = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=4, subsample=0.8,
                           colsample_bytree=0.8, random_state=42)
    xgb_lvl.fit(X_train, y_lvl_train)
    xgb_fr = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=5, subsample=0.8,
                          colsample_bytree=0.8, random_state=42)
    xgb_fr.fit(X_train, y_fr_train)

    def forecast_arima_daily(fit_result, last_date, steps, title):
        fc = fit_result.get_forecast(steps=steps)
        mean = fc.predicted_mean
        ci = fc.conf_int()
        idx = pd.date_range(last_date + pd.Timedelta(days=1), periods=steps, freq="D")
        mean.index = idx
        ci.index = idx
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(mean.index, mean.values, label=f"Forecast {steps} days")
        ax.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], alpha=0.2, label="CI 95%")
        ax.set_title(f"{title} — Forecast {steps} days")
        ax.legend()
        return fig, mean, ci

    def recursive_forecast_xgb_safe(model, X_all, target_base, horizon_days):
        feat_names = getattr(model, "feature_names_in_", X_all.columns)
        X_all = X_all.reindex(columns=feat_names)
        lag_struct = _parse_lag_structure(feat_names)
        work_last = X_all.iloc[-1].copy()
        last_date = X_all.index[-1]
        lag_buffers = {name: [work_last[f"{name}_lag{L}"] for L in Ls] for name, Ls in lag_struct.items() if name in feat_names}
        preds, idxs = [], []
        for h in range(1, horizon_days + 1):
            new_date = last_date + pd.Timedelta(days=h)
            new_row = work_last.copy()
            new_row = _add_fourier_to_row(new_row, new_date)
            new_row = new_row.reindex(feat_names)
            y_hat = model.predict(pd.DataFrame([new_row], index=[new_date]))[0]
            work_last = new_row
            preds.append(y_hat)
            idxs.append(new_date)
        return pd.Series(preds, index=pd.DatetimeIndex(idxs), name=f"{target_base}_xgb_fc")

    def _parse_lag_structure(columns):
        pat = re.compile(r"^(?P<name>.+)_lag(?P<L>\d+)$")
        out = {}
        for c in columns:
            m = pat.match(c)
            if m:
                out.setdefault(m.group("name"), set()).add(int(m.group("L")))
        return {k: sorted(v) for k, v in out.items()}

    def _add_fourier_to_row(row, date):
        if "sin_y" in row.index or "cos_y" in row.index:
            doy = date.dayofyear
            if "sin_y" in row.index: row["sin_y"] = np.sin(2 * np.pi * doy / 365.25)
            if "cos_y" in row.index: row["cos_y"] = np.cos(2 * np.pi * doy / 365.25)
        return row

    forecast_horizon = st.selectbox("Select Forecast Horizon", [15, 30])
    st.subheader(f"ARIMA Forecast — {forecast_horizon} Days")
    fig, _, _ = forecast_arima_daily(model_lvl, lvl.index.max(), forecast_horizon, "Lake_Level ARIMA(1,0,0)")
    st.pyplot(fig)
    fig, _, _ = forecast_arima_daily(model_fr, fr.index.max(), forecast_horizon, "Flow_Rate ARIMA(1,0,1)")
    st.pyplot(fig)

    st.subheader(f"XGBoost Forecast — {forecast_horizon} Days")
    X_all_lvl = pd.concat([X_train, X_test]).sort_index()
    X_all_fr = X_all_lvl.copy()
    fc_lvl = recursive_forecast_xgb_safe(xgb_lvl, X_all_lvl, "Lake_Level", forecast_horizon)
    fc_fr = recursive_forecast_xgb_safe(xgb_fr, X_all_fr, "Flow_Rate", forecast_horizon)
    fig, ax = plt.subplots(figsize=(12, 4))
    y_full_lvl = pd.concat([y_lvl_train, y_lvl_test]).sort_index()
    last_year = y_full_lvl.index.max() - pd.DateOffset(years=1)
    ax.plot(y_full_lvl.loc[last_year:], label="Observed (last year)")
    ax.plot(fc_lvl.index, fc_lvl.values, "o-", label=f"XGB Forecast {forecast_horizon} days")
    ax.set_title(f"Lake_Level — XGBoost Forecast {forecast_horizon} days")
    ax.legend()
    st.pyplot(fig)
    fig, ax = plt.subplots(figsize=(12, 4))
    y_full_fr = pd.concat([y_fr_train, y_fr_test]).sort_index()
    last_year_fr = y_full_fr.index.max() - pd.DateOffset(years=1)
    ax.plot(y_full_fr.loc[last_year_fr:], label="Observed (last year)")
    ax.plot(fc_fr.index, fc_fr.values, "o-", label=f"XGB Forecast {forecast_horizon} days")
    ax.set_title(f"Flow_Rate — XGBoost Forecast {forecast_horizon} days")
    ax.legend()
    st.pyplot(fig)

# Model Comparison
elif section == "Model Comparison":
    st.header("Model Comparison")
    results = {
        "Target": ["Lake_Level", "Lake_Level", "Lake_Level", "Flow_Rate", "Flow_Rate"],
        "Model": ["ARIMA(1,0,0)", "SARIMAX", "XGBRegressor", "ARIMA(1,0,1)", "XGBRegressor"],
        "MAE": [1.9, 1.054, 1.518, 2.9, 2.52],
        "RMSE": [2.45, 1.316, 1.77, 4.4, 4.048]
    }
    df_results = pd.DataFrame(results)
    st.dataframe(df_results.style.set_caption("Model Comparison — Lake_Level and Flow_Rate")
                 .format({"MAE": "{:.3f}", "RMSE": "{:.3f}"})
                 .background_gradient(subset=["MAE", "RMSE"], cmap="Blues"))