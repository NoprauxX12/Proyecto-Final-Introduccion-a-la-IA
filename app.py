import subprocess
import sys
from pathlib import Path

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

# .\.venv\Scripts\python.exe -m streamlit run app.py
if get_script_run_ctx() is None:
    script = Path(__file__).resolve()
    sys.stderr.write(
        "Iniciando Streamlit (la URL aparecera en esta terminal)...\n\n"
    )
    raise SystemExit(
        subprocess.call(
            [sys.executable, "-m", "streamlit", "run", str(script), *sys.argv[1:]]
        )
    )

import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="USD/COP - Predicción con XGBoost",
    page_icon="💵",
    layout="wide"
)

# ── Carga de artefactos ──────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model('model.json')
    return model

@st.cache_data
def load_evals():
    with open('evals_result.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    df = pd.read_csv('datasets/merged_data.csv', parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df

@st.cache_data
def load_raw():
    df = pd.read_csv('datasets/USDCOP_1h_data.csv', sep=';', parse_dates=['Date'])
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.sort_values('Date').reset_index(drop=True)

try:
    model      = load_model()
    evals      = load_evals()
    df_merged  = load_data()
    df_raw     = load_raw()
    datos_ok   = True
except FileNotFoundError:
    datos_ok = False

# ── Encabezado ───────────────────────────────────────────────────────────────

st.title("💵 Predicción del USD/COP con XGBoost")
st.markdown("**Proyecto Final — Introducción a la Inteligencia Artificial**")
st.divider()

if not datos_ok:
    st.error("Primero ejecuta `python USD_COP_prediction.py` para generar los artefactos del modelo.")
    st.stop()

# ── Tabs principales ─────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["📈 Datos históricos", "🧠 Entrenamiento", "📊 Resultados"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATOS HISTÓRICOS
# ════════════════════════════════════════════════════════════════════════════

with tab1:
    st.subheader("Precio USD/COP a lo largo del tiempo")

    col1, col2 = st.columns([3, 1])
    with col2:
        vista = st.selectbox("Vista", ["Último mes", "Últimos 3 meses", "Último año", "Todo"])

    hoy = df_raw['Date'].max()
    if vista == "Último mes":
        df_plot = df_raw[df_raw['Date'] >= hoy - pd.Timedelta(days=30)]
    elif vista == "Últimos 3 meses":
        df_plot = df_raw[df_raw['Date'] >= hoy - pd.Timedelta(days=90)]
    elif vista == "Último año":
        df_plot = df_raw[df_raw['Date'] >= hoy - pd.Timedelta(days=365)]
    else:
        df_plot = df_raw.copy()

    # Calcular medias móviles
    df_plot = df_plot.copy()
    df_plot['SMA20'] = df_plot['Close'].rolling(20).mean()
    df_plot['SMA50'] = df_plot['Close'].rolling(50).mean()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25],
                        vertical_spacing=0.05)

    # Velas japonesas
    fig.add_trace(go.Candlestick(
        x=df_plot['Date'],
        open=df_plot['Open'], high=df_plot['High'],
        low=df_plot['Low'],   close=df_plot['Close'],
        name='USD/COP',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot['SMA20'],
                             line=dict(color='#f4a261', width=1.2),
                             name='SMA 20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot['SMA50'],
                             line=dict(color='#a8dadc', width=1.2),
                             name='SMA 50'), row=1, col=1)

    # RSI
    delta = df_plot['Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - (100 / (1 + rs))

    fig.add_trace(go.Scatter(x=df_plot['Date'], y=rsi,
                             line=dict(color='#b5838d', width=1.2),
                             name='RSI 14'), row=2, col=1)
    fig.add_hline(y=70, line_dash='dash', line_color='#ef5350', row=2, col=1)
    fig.add_hline(y=30, line_dash='dash', line_color='#26a69a', row=2, col=1)

    fig.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        legend=dict(orientation='h', y=1.05),
        margin=dict(t=20, b=20)
    )
    fig.update_yaxes(title_text="Precio (COP)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])

    st.plotly_chart(fig, width="stretch")

    # Métricas rápidas
    ultimo  = df_raw['Close'].iloc[-1]
    anterior = df_raw['Close'].iloc[-2]
    cambio  = (ultimo - anterior) / anterior * 100
    maximo  = df_raw['High'].max()
    minimo  = df_raw['Low'].min()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precio actual",  f"${ultimo:,.2f}",  f"{cambio:+.2f}%")
    m2.metric("Máximo histórico", f"${maximo:,.2f}")
    m3.metric("Mínimo histórico", f"${minimo:,.2f}")
    m4.metric("Total de velas 1H", f"{len(df_raw):,}")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — ENTRENAMIENTO
# ════════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("Curvas de entrenamiento")
    st.markdown("Evolución del **AUC-ROC** en cada iteración del modelo (árbol construido).")

    train_auc = evals['validation_0']['auc']
    val_auc   = evals['validation_1']['auc']
    iters     = list(range(len(train_auc)))

    best_iter = model.best_iteration
    best_val  = val_auc[best_iter]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=iters, y=train_auc,
                              name='Train AUC',
                              line=dict(color='#4cc9f0', width=2)))
    fig2.add_trace(go.Scatter(x=iters, y=val_auc,
                              name='Validation AUC',
                              line=dict(color='#f72585', width=2)))
    fig2.add_vline(x=best_iter, line_dash='dash', line_color='#ffd60a',
                   annotation_text=f"Mejor iteración: {best_iter}  (AUC val: {best_val:.4f})",
                   annotation_position='top right')

    fig2.update_layout(
        height=420,
        template='plotly_dark',
        xaxis_title='Iteración (número de árboles)',
        yaxis_title='AUC-ROC',
        legend=dict(orientation='h', y=1.05),
        margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig2, width="stretch")

    c1, c2, c3 = st.columns(3)
    c1.metric("Mejor iteración", best_iter)
    c2.metric("AUC Train",      f"{train_auc[best_iter]:.4f}")
    c3.metric("AUC Validation", f"{best_val:.4f}")

    st.divider()
    st.subheader("Importancia de variables")

    feature_cols = [c for c in df_merged.columns
                    if c not in ['Date', 'target', 'future_return']]
    importance = pd.Series(model.feature_importances_, index=feature_cols)
    importance = importance.sort_values(ascending=False).head(25)

    colors = []
    for f in importance.index:
        if f.startswith('1h_'):   colors.append('#4cc9f0')
        elif f.startswith('1d_'): colors.append('#f72585')
        elif f.startswith('wti'): colors.append('#f4a261')
        elif f.startswith('dxy'): colors.append('#a8dadc')
        else:                     colors.append('#adb5bd')

    fig3 = go.Figure(go.Bar(
        x=importance.values,
        y=importance.index,
        orientation='h',
        marker_color=colors,
    ))
    fig3.update_layout(
        height=600,
        template='plotly_dark',
        xaxis_title='Importancia',
        yaxis=dict(autorange='reversed'),
        margin=dict(t=10, b=20)
    )
    st.plotly_chart(fig3, width="stretch")

    # Leyenda de colores
    lc1, lc2, lc3, lc4 = st.columns(4)
    lc1.markdown("🔵 Features 1H (horario)")
    lc2.markdown("🔴 Features 1D (diario)")
    lc3.markdown("🟠 WTI (petróleo)")
    lc4.markdown("🩵 DXY (índice dólar)")

    # Importancia por timeframe
    st.divider()
    st.subheader("Importancia por fuente de datos")
    all_imp = pd.Series(model.feature_importances_, index=feature_cols)
    grupos = {
        '1H (técnico horario)': all_imp[[c for c in feature_cols if c.startswith('1h_')]].sum(),
        '1D (técnico diario)':  all_imp[[c for c in feature_cols if c.startswith('1d_')]].sum(),
        'DXY (índice dólar)':   all_imp[[c for c in feature_cols if c.startswith('dxy')]].sum(),
        'WTI (petróleo)':       all_imp[[c for c in feature_cols if c.startswith('wti')]].sum(),
        'Tiempo (hora/día/mes)':all_imp[[c for c in feature_cols if c in ['hour','dayofweek','month']]].sum(),
    }
    df_grupos = pd.DataFrame({'Fuente': list(grupos.keys()), 'Importancia': list(grupos.values())})
    fig4 = px.pie(df_grupos, values='Importancia', names='Fuente',
                  color_discrete_sequence=px.colors.qualitative.Bold,
                  template='plotly_dark')
    fig4.update_layout(height=380, margin=dict(t=10, b=10))
    st.plotly_chart(fig4, width="stretch")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — RESULTADOS
# ════════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("Resultados del modelo en datos no vistos (Test)")

    HORIZON   = 4
    THRESHOLD = 0.3
    split_70  = int(len(df_merged) * 0.70)
    split_85  = int(len(df_merged) * 0.85)

    drop_cols    = ['Date', 'target', 'future_return']
    feature_cols = [c for c in df_merged.columns if c not in drop_cols]

    X = df_merged[feature_cols].values
    y = df_merged['target'].values.astype(int)

    x_val,  y_val  = X[split_70:split_85], y[split_70:split_85]
    x_test, y_test = X[split_85:],          y[split_85:]

    from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

    val_proba  = model.predict_proba(x_val)[:, 1]
    val_pred   = (val_proba >= 0.5).astype(int)
    val_auc_f  = roc_auc_score(y_val, val_proba)
    val_acc_f  = accuracy_score(y_val, val_pred)

    test_proba = model.predict_proba(x_test)[:, 1]
    test_pred  = (test_proba >= 0.5).astype(int)
    test_auc   = roc_auc_score(y_test, test_proba)
    test_acc   = accuracy_score(y_test, test_pred)
    cm         = confusion_matrix(y_test, test_pred)

    # Métricas
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AUC-ROC Validation", f"{val_auc_f:.4f}")
    col2.metric("Accuracy Validation", f"{val_acc_f*100:.2f}%")
    col3.metric("AUC-ROC Test",  f"{test_auc:.4f}")
    col4.metric("Accuracy Test", f"{test_acc*100:.2f}%")

    st.divider()

    izq, der = st.columns(2)

    with izq:
        st.markdown("#### Matriz de confusión (Test)")
        cm_df = pd.DataFrame(cm,
                             index=['Real: Baja', 'Real: Sube'],
                             columns=['Pred: Baja', 'Pred: Sube'])
        fig5 = px.imshow(cm_df, text_auto=True,
                         color_continuous_scale='Blues',
                         template='plotly_dark',
                         aspect='auto')
        fig5.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig5, width="stretch")

    with der:
        st.markdown("#### Interpretación")
        tn, fp, fn, tp = cm.ravel()
        precision_sube = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_sube    = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision_baja = tn / (tn + fn) if (tn + fn) > 0 else 0
        recall_baja    = tn / (tn + fp) if (tn + fp) > 0 else 0

        st.markdown(f"""
| | Precisión | Recall |
|---|---|---|
| **Sube** | {precision_sube*100:.1f}% | {recall_sube*100:.1f}% |
| **Baja** | {precision_baja*100:.1f}% | {recall_baja*100:.1f}% |
        """)
        st.info(f"""
**Total predicciones test:** {len(y_test)}

✅ Correctas: {tp + tn} ({(tp+tn)/len(y_test)*100:.1f}%)

❌ Incorrectas: {fp + fn} ({(fp+fn)/len(y_test)*100:.1f}%)
        """)

    st.divider()
    st.subheader("Probabilidades predichas en el conjunto Test")
    st.markdown("Cada punto es una vela horaria. El eje Y muestra la probabilidad de que el precio **suba** en las próximas 4 horas.")

    df_test_plot = df_merged.iloc[split_85:].copy()
    df_test_plot['prob_sube'] = test_proba
    df_test_plot['correcto']  = (test_pred == y_test)
    df_test_plot['etiqueta']  = df_test_plot['correcto'].map({True: 'Correcto', False: 'Error'})

    fig6 = px.scatter(
        df_test_plot, x='Date', y='prob_sube',
        color='etiqueta',
        color_discrete_map={'Correcto': '#26a69a', 'Error': '#ef5350'},
        opacity=0.7,
        template='plotly_dark',
        labels={'prob_sube': 'P(Sube)', 'Date': 'Fecha'}
    )
    fig6.add_hline(y=0.5, line_dash='dash', line_color='white',
                   annotation_text='Umbral 0.5')
    fig6.update_layout(height=380, margin=dict(t=10, b=10))
    st.plotly_chart(fig6, width="stretch")

st.divider()
st.caption("Juan Sebastián Lizcano Urrea · Jose Andres Mendoza Hernandez — Proyecto Final IA 2026")
