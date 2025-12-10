# app.py — Versión Profesional / Estilo Empresarial
"""
Frontend Empresarial en Streamlit
Interfaz Corporativa para un Chatbot Analítico con integración hacia API FastAPI.
Optimizado para entornos productivos empresariales.
"""

import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd
from typing import Optional, Dict, Any
import uuid
import os

# =============================================
#  🔐 Session Management — ID Persistente
# =============================================
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

API_URL = os.getenv("API_URL", "http://app:8000")  # Docker: app | Local: localhost

# =============================================
#  🖥️ Configuración Global de la App
# =============================================

st.set_page_config(
    page_title="Analytical Chatbot Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================
#  🛠️ Funciones Utilitarias
# =============================================

def check_api_health() -> bool:
    """Valida conexión con la API Backend."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def send_message(message: str, conversation_id: Optional[str] = None) -> Dict[str, Any]:
    """Envía un mensaje al endpoint /chat de la API empresarial."""
    try:
        payload = {"message": message}
        if conversation_id:
            payload["conversation_id"] = conversation_id

        response = requests.post(
            f"{API_URL}/chat",
            json=payload,
            timeout=60,
        )

        if response.status_code == 200:
            return response.json()
        else:
            return {
                "error": f"Error {response.status_code}: {response.text}",
                "response": "Error procesando la solicitud en el backend."
            }

    except requests.exceptions.Timeout:
        return {
            "error": "Timeout: El backend tardó demasiado.",
            "response": "La respuesta está tardando. Inténtalo nuevamente."
        }

    except Exception as e:
        return {
            "error": str(e),
            "response": "No fue posible establecer comunicación con la API."
        }


def render_chart(chart_config: Dict[str, Any]):
    """Renderiza un gráfico Plotly basado en configuración JSON."""
    try:
        fig = go.Figure(chart_config.get("config", {}))
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"No se pudo renderizar el gráfico: {e}")


def render_table(results: list):
    if results:
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)


def render_kpis(kpis: Dict[str, Any]):
    if not kpis:
        return

    cols = st.columns(min(len(kpis), 4))

    for idx, (key, k) in enumerate(kpis.items()):
        with cols[idx % 4]:
            st.metric(
                label=k.get("name", key),
                value=k.get("formatted_value", "N/A"),
                help=k.get("description", "")
            )

# =============================================
#  📌 SIDEBAR CORPORATIVO
# =============================================

with st.sidebar:
    st.title("⚙️ Panel de Control")

    # Estado de API
    if check_api_health():
        st.success("🟢 API operativa")
    else:
        st.error("🔴 API no disponible")
        st.info(f"Endpoint: {API_URL}")

    st.divider()

    show_raw_response = st.checkbox("Mostrar respuesta JSON", value=False)
    show_sql = st.checkbox("Mostrar SQL generado", value=True)

    st.divider()

    st.subheader("📝 Consultas Ejemplo")

    examples = [
        "¿Cuántas ventas preventivas hay?",
        "Muéstrame los 10 productos más vendidos",
        "Calcula el revenue total",
        "Gráfica de ventas por mes",
        "Compara preventivas vs correctivas",
        "¿Cuál es el ticket promedio?"
    ]

    for example in examples:
        if st.button(example, key=f"ex_{example}", use_container_width=True):
            st.session_state.example_query = example

    st.divider()

    if st.button("🗑️ Limpiar caché API", use_container_width=True):
        try:
            res = requests.delete(f"{API_URL}/cache")
            st.success("Caché limpiado correctamente" if res.status_code == 200 else "No se pudo limpiar caché")
        except Exception:
            st.error("Conexión fallida con el backend")

    st.markdown("---")
    st.subheader("📊 Métricas Operacionales")

    if st.button("Ver Métricas", use_container_width=True):
        try:
            metrics = requests.get(f"{API_URL}/metrics?days=7", timeout=10).json()

            st.markdown("## 🧠 Indicadores de la Plataforma (7 días)")

            gen = metrics.get("general", {})
            c1, c2, c3 = st.columns(3)
            c1.metric("Interacciones", gen.get("total_interactions", 0))
            c2.metric("Rating Promedio", f"{gen.get('avg_rating', 0)}/5.0")
            c3.metric("Tiempo Respuesta", f"{gen.get('avg_response_time_ms', 0)} ms")

            if metrics.get("rating_distribution"):
                st.bar_chart(pd.DataFrame([
                    {"Rating": f"{k}⭐", "Cantidad": v}
                    for k, v in metrics["rating_distribution"].items()
                ]).set_index("Rating"))

            if metrics.get("top_errors"):
                st.markdown("### Errores más frecuentes")
                st.dataframe(pd.DataFrame(metrics["top_errors"]))

        except Exception as e:
            st.error(f"No se pudieron obtener métricas: {e}")

# =============================================
#  🧠 MAIN — Chat Analítico Empresarial
# =============================================

st.title("Analytical Chatbot Platform")
st.markdown("Interactúa con tus datos empresariales usando **lenguaje natural**.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Historial
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg.get("chart_config"):
            render_chart(msg["chart_config"])

        if msg.get("results"):
            with st.expander("📋 Ver datos completos"):
                render_table(msg["results"])

        if msg.get("kpis"):
            st.markdown("### KPIs")
            render_kpis(msg["kpis"])

        if msg.get("sql_query") and show_sql:
            with st.expander("🧬 SQL Generado"):
                st.code(msg["sql_query"], language="sql")

# =============================================
#  ✏️ Entrada del usuario
# =============================================

user_input = st.chat_input("Escribe tu consulta empresarial...")

if "example_query" in st.session_state:
    user_input = st.session_state.example_query
    del st.session_state.example_query

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Procesando información..."):

            response = send_message(user_input)

            st.markdown(response.get("response", "Sin respuesta."))

            if response.get("chart_config"):
                render_chart(response["chart_config"])

            if response.get("results"):
                with st.expander("📋 Ver tabla completa"):
                    render_table(response["results"])

            if response.get("kpis"):
                st.markdown("### KPIs")
                render_kpis(response["kpis"])

            if response.get("sql_query") and show_sql:
                with st.expander("🧬 SQL"):
                    st.code(response["sql_query"], language="sql")

            if show_raw_response:
                with st.expander("📄 RAW JSON"):
                    st.json(response)

            if response.get("error"):
                st.error(f"⚠️ {response['error']}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": response.get("response", "Sin respuesta"),
                "chart_config": response.get("chart_config"),
                "results": response.get("results"),
                "kpis": response.get("kpis"),
                "sql_query": response.get("sql_query"),
            })

# =============================================
#  🦶 FOOTER
# =============================================

st.divider()
st.markdown("""
<div style='text-align:center; color:gray; font-size:0.8em;'>
 💡 Consejos: solicita gráficas, KPIs, SQL o análisis avanzados.<br>
 Ejemplos: "Ventas por mes" • "Revenue total" • "Top productos vendidos".
</div>
""", unsafe_allow_html=True)
