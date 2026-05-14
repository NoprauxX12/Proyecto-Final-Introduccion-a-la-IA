import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score, confusion_matrix
)


HORIZON   = 4      # Velas de 1H a predecir (4h adelante)
THRESHOLD = 0.3    # Cambio mínimo % para considerar operación válida


def calc_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def add_features(df, prefix=''):
    c = df['Close']
    p = prefix

    out = pd.DataFrame(index=df.index)

    for n in [1, 2, 3, 5, 10, 20]:
        out[f'{p}ret_{n}'] = c.pct_change(n) * 100
    out[f'{p}range_pct']      = (df['High'] - df['Low']) / c * 100
    out[f'{p}body_pct']       = (c - df['Open']) / c * 100
    out[f'{p}upper_wick_pct'] = (df['High'] - df[['Open','Close']].max(axis=1)) / c * 100
    out[f'{p}lower_wick_pct'] = (df[['Open','Close']].min(axis=1) - df['Low']) / c * 100

    # Cruces de medias móviles
    for w in [10, 20, 50, 100]:
        sma = c.rolling(w).mean()
        ema = c.ewm(span=w, adjust=False).mean()
        out[f'{p}dist_sma_{w}'] = (c - sma) / sma * 100
        out[f'{p}dist_ema_{w}'] = (c - ema) / ema * 100

    # Cruces de medias móviles
    sma10 = c.rolling(10).mean()
    sma20 = c.rolling(20).mean()
    sma50 = c.rolling(50).mean()
    out[f'{p}cross_10_20'] = (sma10 - sma20) / sma20 * 100
    out[f'{p}cross_20_50'] = (sma20 - sma50) / sma50 * 100

    # RSI revisión si el precio está por encima o por debajo de la media
    out[f'{p}rsi_7']  = calc_rsi(c, 7)
    out[f'{p}rsi_14'] = calc_rsi(c, 14)
    out[f'{p}rsi_28'] = calc_rsi(c, 28)

    # MACD revisión si el movimiento es fuerte o débil
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    sig   = macd.ewm(span=9, adjust=False).mean()
    out[f'{p}macd']      = macd / c * 100
    out[f'{p}macd_sig']  = sig  / c * 100
    out[f'{p}macd_hist'] = (macd - sig) / c * 100

    # Bollinger Bands revisión si el precio está por encima o por debajo de la media
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    bb_up  = bb_mid + 2 * bb_std
    bb_dn  = bb_mid - 2 * bb_std
    out[f'{p}bb_width']    = (bb_up - bb_dn) / bb_mid * 100
    out[f'{p}bb_position'] = (c - bb_dn) / (bb_up - bb_dn).replace(0, np.nan)

    # ATR revisión si el precio está por encima o por debajo de la media
    pc  = c.shift(1)
    tr  = np.maximum(df['High']-df['Low'],
          np.maximum(abs(df['High']-pc), abs(df['Low']-pc)))
    out[f'{p}atr_14'] = tr.rolling(14).mean() / c * 100
    out[f'{p}atr_20'] = tr.rolling(20).mean() / c * 100

    # Revisión de la estocástica si el precio está por encima o por debajo de la media
    low14  = df['Low'].rolling(14).min()
    high14 = df['High'].rolling(14).max()
    stoch  = (c - low14) / (high14 - low14).replace(0, np.nan) * 100
    out[f'{p}stoch_k'] = stoch
    out[f'{p}stoch_d'] = stoch.rolling(3).mean()

    out[f'{p}mom_10'] = c.pct_change(10) * 100
    out[f'{p}mom_20'] = c.pct_change(20) * 100

    return out


def load_and_prepare(path):
    df = pd.read_csv(path, sep=';')
    df['Date'] = pd.to_datetime(df['Date'])
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


"""
    Load data stage
"""

df_1h = load_and_prepare('datasets/USDCOP_1h_data.csv')
df_1d = load_and_prepare('datasets/USDCOP_1d_data.csv')
df_wti = pd.read_csv('datasets/WTI_1d_data.csv', sep=';', parse_dates=['Date'])
df_dxy = pd.read_csv('datasets/DXY_1d_data.csv', sep=';', parse_dates=['Date'])

print(f"1H: {len(df_1h)} filas | Rango: {df_1h['Date'].min().date()} - {df_1h['Date'].max().date()}")
print(f"1D: {len(df_1d)} filas | Rango: {df_1d['Date'].min().date()} - {df_1d['Date'].max().date()}")


# Calcular features por timeframe
feat_1h = add_features(df_1h, prefix='1h_')
feat_1d = add_features(df_1d, prefix='1d_')

feat_1h['Date'] = df_1h['Date']
# Desplazamos 1 día para que cada vela horaria solo vea el cierre del día anterior
feat_1d['Date'] = df_1d['Date'] + pd.Timedelta(days=1)

# Target definido sobre 1H, si el retorno es mayor a THRESHOLD se toma 1, si es menor a -THRESHOLD se toma 0
df_1h['future_return'] = df_1h['Close'].pct_change(HORIZON).shift(-HORIZON) * 100
df_1h['target'] = np.where(
    df_1h['future_return'] >  THRESHOLD, 1,
    np.where(
    df_1h['future_return'] < -THRESHOLD, 0,
    np.nan)
)
feat_1h['future_return'] = df_1h['future_return']
feat_1h['target']        = df_1h['target']

# Merge 1D features sobre 1H (contexto diario en cada vela horaria)
feat_1h = feat_1h.dropna(subset=['Date']).sort_values('Date')
feat_1d = feat_1d.dropna(subset=['Date']).sort_values('Date')
df_merged = pd.merge_asof(feat_1h, feat_1d, on='Date', direction='backward')

# Merge WTI y DXY como contexto macroeconómico
df_wti = df_wti.sort_values('Date')
df_dxy = df_dxy.sort_values('Date')
df_merged = pd.merge_asof(df_merged, df_wti[['Date', 'WTI']], on='Date', direction='backward')
df_merged = pd.merge_asof(df_merged, df_dxy[['Date', 'DXY']], on='Date', direction='backward')

# Retornos de WTI y DXY
df_merged['wti_ret_1']  = df_merged['WTI'].pct_change(1) * 100
df_merged['wti_ret_5']  = df_merged['WTI'].pct_change(5) * 100
df_merged['wti_ret_20'] = df_merged['WTI'].pct_change(20) * 100
df_merged['dxy_ret_1']  = df_merged['DXY'].pct_change(1) * 100
df_merged['dxy_ret_5']  = df_merged['DXY'].pct_change(5) * 100
df_merged['dxy_ret_20'] = df_merged['DXY'].pct_change(20) * 100
df_merged.drop(columns=['WTI', 'DXY'], inplace=True)

df_merged['hour']      = df_merged['Date'].dt.hour
df_merged['dayofweek'] = df_merged['Date'].dt.dayofweek
df_merged['month']     = df_merged['Date'].dt.month

df_merged.dropna(inplace=True)
df_merged.reset_index(drop=True, inplace=True)

drop_cols    = ['Date', 'target', 'future_return']
feature_cols = [c for c in df_merged.columns if c not in drop_cols]

X = df_merged[feature_cols].values
y = df_merged['target'].values.astype(int)

# Split de datos para entrenamiento, validación y prueba
train_end = int(len(X) * 0.70)
val_end   = int(len(X) * 0.85)

x_train, y_train = X[:train_end],        y[:train_end]
x_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
x_test,  y_test  = X[val_end:],          y[val_end:]

print(f"\nTrain:      {len(x_train)} muestras")
print(f"Validation: {len(x_val)} muestras")
print(f"Test:       {len(x_test)} muestras")

# Balance de clases, se calcula el número de positivos y negativos
n_pos = y_train.sum()
n_neg = len(y_train) - n_pos
scale_pos = n_neg / n_pos

# Entrenamiento del modelo XGBoost
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    max_depth=4,
    learning_rate=0.02,
    n_estimators=2000,
    alpha=8,
    reg_lambda=8,
    colsample_bytree=0.7,
    subsample=0.8,
    min_child_weight=20,
    scale_pos_weight=scale_pos,
    early_stopping_rounds=100,
    n_jobs=-1,
    random_state=123,
)

# Entrenamiento del modelo XGBoost
print("\nEntrenando...")
model.fit(
    x_train, y_train,
    eval_set=[(x_train, y_train), (x_val, y_val)],
    verbose=200,
)

print(f"\nMejor iteracion: {model.best_iteration}")

# Guardar artefactos para la interfaz web
import pickle
model.save_model('model.json')
with open('evals_result.pkl', 'wb') as f:
    pickle.dump(model.evals_result(), f)
df_merged.to_csv('datasets/merged_data.csv', index=False)
print("Artefactos guardados.")

# Evaluación en Validation
val_proba = model.predict_proba(x_val)[:, 1]
val_pred  = (val_proba >= 0.5).astype(int)
val_auc   = roc_auc_score(y_val, val_proba)
val_acc   = accuracy_score(y_val, val_pred)

# Evaluación en Test
y_pred_proba = model.predict_proba(x_test)[:, 1]
y_pred       = (y_pred_proba >= 0.5).astype(int)
auc      = roc_auc_score(y_test, y_pred_proba)
accuracy = accuracy_score(y_test, y_pred)
cm       = confusion_matrix(y_test, y_pred)


print("RESULTADOS - USD/COP")
print(f"{'Conjunto':<15} {'AUC-ROC':>10} {'Accuracy':>10}")
print(f"{'Train':15} {'(visto)':>10} {'(visto)':>10}")
print(f"{'Validation':<15} {val_auc:>10.4f} {val_acc*100:>9.2f}%")
print(f"{'Test':<15} {auc:>10.4f} {accuracy*100:>9.2f}%")
print()
print("Matriz de confusion (Test):")
print(f"              Pred Baja   Pred Sube")
print(f"  Real Baja      {cm[0][0]:>6}      {cm[0][1]:>6}")
print(f"  Real Sube      {cm[1][0]:>6}      {cm[1][1]:>6}")

importance = pd.Series(model.feature_importances_, index=feature_cols)
importance = importance.sort_values(ascending=False)

print(f"\nFeatures mas importantes:")
print(importance.head(20).to_string())

print("\n--- Features por timeframe ---")
for pref in ['1h_', '1d_', 'wti_', 'dxy_']:
    cols = [c for c in feature_cols if c.startswith(pref)]
    imp  = importance[cols].sum()
    print(f"  {pref:<6} importancia total: {imp:.4f}  ({imp*100:.1f}%)")
