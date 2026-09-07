import numpy as np
import streamlit as st
import plotly.graph_objects as go

def render(results: dict) -> None:
    anomaly_score = results["anomaly_score"]
    labels = results["labels"]
    auto_threshold = results["auto_threshold"]
    train_stats = results["train_stats"]
    T = len(anomaly_score)
    time_idx = np.arange(T)

    st.subheader("Widok globalny: wynik anomalii w czasie")

    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("Liczba okien", f"{T:,}")
    
    method = train_stats.get("method", "Standard")
    if method == "POT_GPD":
        threshold_help = (
        "Próg detekcji wyznaczony metodą Peak Over Threshold "
        "z dopasowaniem rozkładu GPD do skrajnych wartości "
        "treningowego wyniku anomalii. "
        f"Próg bazowy POT: {train_stats.get('base_t', 0):.5f}, "
        f"risk_q: {train_stats.get('risk_q', 0.001)}."
    )
    else:
        threshold_help = (
        "Próg detekcji wyznaczony na podstawie statystyk "
        "błędu treningowego. "
        f"Średnia: {train_stats.get('mean', 0):.4f}, "
        f"odchylenie standardowe: {train_stats.get('std', 0):.4f}."
    )
        
    col_info2.metric("Próg detekcji", f"{auto_threshold:.4f}", help=threshold_help)

    anomaly_pct = 100 * (labels == 1).mean()
    col_info3.metric("Udział oznaczonych anomalii", f"{anomaly_pct:.1f}%")

    st.divider()

    min_slider_val = float(np.min(anomaly_score))
    max_slider_val = float(max(np.max(anomaly_score), auto_threshold) * 1.05)

    if "what_if_threshold" not in st.session_state:
        st.session_state["what_if_threshold"] = float(auto_threshold)
    else:
        if st.session_state["what_if_threshold"] > max_slider_val:
            st.session_state["what_if_threshold"] = max_slider_val
        elif st.session_state["what_if_threshold"] < min_slider_val:
            st.session_state["what_if_threshold"] = min_slider_val

    if "top_k_param" not in st.session_state:
        st.session_state["top_k_param"] = 5 # Domyślnie SOTA

    st.markdown("### Analiza scenariuszowa")
    st.caption("Sprawdź, jak zmiana progu detekcji oraz liczby cech uwzględnianych w agregacji "
               "wpływa na czułość modelu i liczbę generowanych alarmów."
               )
    
    col_slider, col_k, col_btn = st.columns([4, 2, 1])

    with col_k:
        top_k_val = st.slider(
            "Liczba cech w agregacji Top-K",
            min_value=1,
            max_value=38,
            value=st.session_state["top_k_param"],
            step=1,
            key="top_k_param_slider",
            help=(
            "Wynik anomalii jest wyznaczany na podstawie średniego błędu MAE "
            "dla K cech o największym odchyleniu. "
            "Mniejsze wartości eksponują lokalne odchylenia, a większe "
            "uwzględniają szerszy kontekst wszystkich cech."
        ),
        )
        st.session_state["top_k_param"] = top_k_val

    with col_btn:
        st.write("")
        st.write("") 
        if st.button("🔄 Reset progu", help="Przywraca domyślny próg detekcji wyznaczony dla wybranego serwera."):
            st.session_state["what_if_threshold"] = float(auto_threshold)
            st.rerun()

    with col_slider:
        threshold = st.slider(
            "Próg detekcji",
            min_value = min_slider_val,
            max_value = max_slider_val,
            step = float((max_slider_val - min_slider_val) / 500),
            key = "what_if_threshold",
            help=(
            "Niższy próg zwiększa czułość detekcji i liczbę wykrywanych odchyleń, "
            "natomiast wyższy próg ogranicza liczbę alarmów."
            ),
        )
    
    st.session_state["global_threshold"] = threshold
    predictions = (anomaly_score >= threshold).astype(int)

    with st.container():
        fig = go.Figure()
        
        in_anomaly = False
        start_idx = 0
        for i in range(T):
            if labels[i] == 1 and not in_anomaly:
                in_anomaly = True
                start_idx = i
            elif labels[i] == 0 and in_anomaly:
                in_anomaly = False
                fig.add_vrect(x0=start_idx, x1=i, fillcolor="rgba(161,44,123,0.15)", layer="below", line_width=0)
        if in_anomaly:
            fig.add_vrect(x0=start_idx, x1=T-1, fillcolor="rgba(161,44,123,0.15)", layer="below", line_width=0)

        fig.add_trace(go.Scatter(
            x=time_idx, y=anomaly_score,
            mode="lines",
            name="Zagregowany błąd L1",
            line=dict(color="#01696f", width = 1),
            hovertemplate="Okno=%{x}<br>Wartość=%{y:.5f}<extra></extra>",
        ))

        det_idx = time_idx[predictions == 1]
        det_val = anomaly_score[predictions == 1]
        if len(det_idx):
            fig.add_trace(go.Scatter(
                x=det_idx, y=det_val,
                mode="markers",
                name="Wykryte anomalie",
                marker=dict(color="#a12c7b", size=4, opacity=0.8),
                hovertemplate="Okno=%{x}<br>Wartość=%{y:.5f}<extra></extra>",
            ))

        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color = "#a13544",
            annotation_text = f"Próg detekcji: {threshold:.4f}",
            annotation_position = "top right",
        )

        fig.update_layout(
            height = 360,
            margin = dict(l=10, r=10, t=10, b=40),
            legend = dict(orientation="h", y=-0.15),
            xaxis_title = "Indeks okna czasowego",
            yaxis_title = f"Wartość błędu L1 dla Top-{top_k_val} cech",
            plot_bgcolor = "rgba(0,0,0,0)",
            paper_bgcolor = "rgba(0,0,0,0)",
        )
        fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})