import streamlit as st
import torch

from src.app.helpers import (
    load_config,
    list_machines,
    load_pretrained,
    run_inference,
    machine_display_name,
    model_display_name,
)
import src.app.views.overview as view_overview
import src.app.views.attribution as view_attribution
import src.app.views.detail as view_detail


st.set_page_config(
    page_title="System detekcji anomalii w SMD",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


config = load_config()
available_machines = list_machines(config)


with st.sidebar:
    st.title("🔍 SMD Anomaly Analyzer")
    st.caption("System detekcji anomalii w wielowymiarowych szeregach czasowych dedykowany dla zbioru TSB-AD SMD")
    st.divider()

    st.markdown("### Wybór maszyny")

    if not available_machines:
        st.error(
            "Brak wytrenowanych modeli. "
            "Uruchom `python scripts/train_all.py`, aby wytrenować modele dla wszystkich maszyn."
        )
        st.stop()

    machine_labels = {stem: machine_display_name(stem) for stem in available_machines}
    model_labels = {stem: model_display_name(stem) for stem in available_machines}

    selected_stem = st.selectbox(
        "Serwer",
        options=available_machines,
        format_func=lambda s: machine_labels[s],
        help=(
            "Wybierz serwer do analizy. "
            "Dla każdego serwera dostępny jest dedykowany model LSTM-AE-V9 "
            "wytrenowany na danych reprezentujących jego normalne działanie."
        ),
    )

    st.divider()

    st.markdown("### Dane do analizy")
    uploaded_test = st.file_uploader(
        "Wgraj plik CSV",
        type=["csv"],
        help=(
            "Plik CSV z danymi do analizy (38 cech + opcjonalna kolumna etykiet). "
            "Może to być dowolny fragment sygnału z wybranej maszyny."
        ),
    )

    st.divider()

    st.markdown("**Hierarchia widoków**")
    st.markdown(
        """
        1. 🌍 **Globalny** — wynik anomalii dla całego przebiegu sygnału
        2. 🗺️ **Atrybucja** — mapa cieplna błędów dla wszystkich cech
        3. 🔬 **Szczegółowy** — sygnał vs rekonstrukcja dla wybranej oraz procent udziału w błędzie
        """
    )


with st.spinner(f"Ładowanie modelu dla: {machine_labels[selected_stem]}..."):
    try:
        model, scaler, threshold, train_stats = load_pretrained(
            selected_stem, config
        )
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

st.sidebar.success(f"✅ Model załadowany: {model_labels[selected_stem]}")


if uploaded_test is None:
    st.info(
        "Wybierz maszynę i wgraj plik CSV z danymi do analizy w panelu bocznym."
    )

    st.subheader(f"Załadowany model: {model_labels[selected_stem]}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Serwer", machine_labels[selected_stem])
    col2.metric("Próg detekcji", f"{threshold:.5f}",
                 help=(
                     "Wartość progowa wyznaczona metodą Peak Over Threshold "
                     "z dopasowaniem rozkładu GPD do skrajnych wartości "
                     "treningowego wyniku anomalii."
                    ),
                )
    if train_stats:
        col3.metric(
            "Próg bazowy POT",
            # f"{train_stats.get('mean', 0):.4f} ± {train_stats.get('std', 0):.4f}",
            f"{train_stats.get('base_threshold', 0)}",
            help=(
                "Wartość bazowa wyznaczona jako wysoki percentyl treningowego wyniku anomalii. "
                "Przekroczenia ponad ten poziom są wykorzystywane do dopasowania modelu ogona GPD."
            ),
        )
    st.stop()

with st.spinner("Ładowanie..."):
    try:
        results, raw_test_data = run_inference(
            _model=model,
            _device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            _scaler=scaler,
            file_bytes_test=uploaded_test.read(),
            filename=uploaded_test.name,
            config=config,
            threshold=threshold,
            train_stats=train_stats,
            top_k=st.session_state.get("top_k_param", 5),
        )
    except Exception as e:
        st.error(f"Błąd podczas analizy pliku: {e}")
        st.stop()


tab1, tab2, tab3 = st.tabs([
    "🌍 Widok globalny",
    "🗺️ Widok atrybucji",
    "🔬 Widok szczegółowy",
])

with tab1:
    view_overview.render(results)

with tab2:
    view_attribution.render(results)

with tab3:
    view_detail.render(results, raw_test_data)