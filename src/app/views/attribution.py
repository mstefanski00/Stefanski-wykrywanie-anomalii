import numpy as np
import streamlit as st
import plotly.graph_objects as go 

def render(results: dict) -> None:
    error_matrix = results["error_matrix"]
    T, F = error_matrix.shape

    st.subheader("🗺️ Widok atrybucji: Mapa błędów rekonstrukcji")
    st.caption(
    "Mapa przedstawia wartości błędu rekonstrukcji L1 dla poszczególnych cech w kolejnych oknach czasowych. "
    "Jaśniejsze obszary wskazują cechy, które w danym fragmencie najsilniej odbiegają od wzorca normalnego działania. "
    "Użyj suwaka, aby ograniczyć analizowany zakres czasu."
    )
    
    time_range = st.slider(
        "Zakres okien czasowych",
        min_value=0,
        max_value=T-1,
        value=(0, min(T-1, 3000)),
        step=1,
        key="heatmap_range",
        help="Ogranicz widoczny zakres osi czasu, aby ułatwić analizę i przyspieszyć renderowanie mapy."
    )

    t0, t1 = time_range
    error_slice = error_matrix[t0:t1+1, :].T 

    feature_ids = [f"F{i:02d}" for i in range(1, F + 1)]

    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        z=error_slice,
        x=np.arange(t0, t1+1),
        y=feature_ids,
        colorscale="Inferno",
        name="Błąd rekonstrukcji L1",
        hovertemplate="Okno=%{x}<br>Cecha=%{y}<br>Błąd L1=%{z:.4f}<extra></extra>",
        showscale=True,
        colorbar=dict(title="Błąd L1", thickness=15)
    ))

    fig.update_layout(
        height=500,
        margin=dict(l=10, r=10, t=10, b=40),
        xaxis_title="Indeks okna czasowego",
        yaxis_title="ID Cechy",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(
            tickmode='array',
            tickvals=feature_ids,
            ticktext=feature_ids,
            autorange='reversed'
        )
    )

    event = st.plotly_chart(
        fig,
        use_container_width=True,
        on_select="rerun",
        key="heatmap_chart",
        config={"displayModeBar": False}
    )

    if event and event.get("selection") and event["selection"].get("points"):
        pt = event["selection"]["points"][0]
        clicked_feat_label = pt.get("y", "F01")
        try:
            clicked_feat_idx = int(clicked_feat_label.replace("F", "")) - 1
            st.session_state["selected_feature"] = clicked_feat_idx
            st.toast(f"Wybrano cechę F{clicked_feat_idx:02d}! Przejdź teraz do zakładki 'Widok szczegółowy'.")
            st.rerun()
        except ValueError:
            pass