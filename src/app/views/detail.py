import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def render(results: dict, raw_data: dict) -> None:
    error_matrix = results["error_matrix"]
    reconstructed = results["reconstructed"]
    labels = results["labels"]
    T, F = error_matrix.shape
    time_idx = np.arange(T)

    if "selected_feature" not in st.session_state:
        st.session_state["selected_feature"] = 0

    st.subheader("🔬 Widok szczegółowy: sygnał, rekonstrukcja i błąd")
    
    feature_ids = [f"F{i:02d}" for i in range(1, F+1)]
    current_selection = int(st.session_state["selected_feature"])
    
    selected_feat_label = st.selectbox(
        "Wybierz cechę do szczegółowej analizy:",
        options=feature_ids,
        index=current_selection if current_selection < F else 0,
        help=(
        "Wybierz cechę, aby porównać przebieg sygnału rzeczywistego, "
        "rekonstrukcji modelu oraz odpowiadającego im błędu."
    ),
    )
    
    feat_idx = int(selected_feat_label.replace("F", "")) - 1
    st.session_state["selected_feature"] = feat_idx

    raw_test = raw_data.get("raw_test")
    window_size = results["window_size"]
    last_indices = np.arange(window_size - 1, window_size - 1 + T)
    
    if last_indices[-1] < len(raw_test):
        signal_raw = raw_test[last_indices, feat_idx]
    else:
        signal_raw = np.full(T, np.nan)
        
    recon_feat = reconstructed[:, feat_idx]
    error_feat = error_matrix[:, feat_idx]

    total_error_per_step = error_matrix.sum(axis=1) + 1e-9
    contribution_pct = (error_feat / total_error_per_step) * 100

    st.caption(
        f"Analiza cechy **F{feat_idx + 1:02d}**. "
        "Górny wykres przedstawia porównanie rzeczywistego przebiegu sygnału z rekonstrukcją modelu. "
        "Dolny wykres pokazuje procentowy udział czujnika w całkowitym błędzie wszystkich 38 cech w analizowanym przedziale czasu."
    )

    time_range = st.slider(
        "Zakres okien czasowych",
        min_value=0, max_value=T-1,
        value=(0, min(T-1, 1500)),
        step=1, key="detail_range",
        help="Ogranicz widoczny zakres czasu, aby ułatwić szczegółową analizę przebiegu."
    )
    
    t0, t1 = time_range
    t_slice = time_idx[t0:t1+1]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.65, 0.35]
    )

    fig.add_trace(go.Scatter(
        x=t_slice, y=signal_raw[t0:t1+1],
        mode="lines", name="Sygnał rzeczywisty",
        line=dict(color="#006494", width=2),
        hovertemplate="Okno=%{x}<br>Sygnał=%{y:.5f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=t_slice, y=recon_feat[t0:t1+1],
        mode="lines", name="Rekonstrukcja modelu",
        line=dict(color="#e2711d", width=2, dash="dash"),
        hovertemplate="Okno=%{x}<br>Rekonstrukcja=%{y:.5f}<extra></extra>",
    ), row=1, col=1)

    # Wykres 2: Procentowy udział w błędzie
    contribution_slice = contribution_pct[t0:t1+1]
    # Słupki zmieniają kolor na czerwony, gdy wkład czujnika w awarię przekracza 20%
    bar_colors = np.where(contribution_slice > 20, "#a13544", "#01696f")

    fig.add_trace(go.Bar(
        x=t_slice, y=contribution_slice,
        name="Procentowy udział cechy w błędzie L1",
        marker_color=bar_colors,
        hovertemplate="Okno=%{x}<br>Udział=%{y:.1f}%<extra></extra>",
    ), row=2, col=1)

    fig.update_layout(
        height=550,
        margin=dict(l=10, r=10, t=10, b=40),
        legend=dict(orientation="h", y=-0.15),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        bargap=0.1,
        xaxis2_title="Indeks okna czasowego",
    )
    
    fig.update_yaxes(title_text="Wartości sygnału", row=1, col=1, showgrid=True, gridcolor="rgba(0,0,0,0.05)")
    fig.update_yaxes(title_text="Udział w procentach", range=[0, 105], row=2, col=1, showgrid=True, gridcolor="rgba(0,0,0,0.05)")

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})