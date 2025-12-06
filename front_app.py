"""
Frontend Streamlit para Chatbot Analítico
Consume la API FastAPI y renderiza gráficos interactivos
"""

import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd
from typing import Optional, Dict, Any
import json

import uuid

# Generar session_id persistente
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# ============ Configuración ============

import os
API_URL = os.getenv("API_URL", "http://app:8000")  # En Docker usa 'app', localmente 'localhost'

st.set_page_config(
    page_title="Chatbot Analítico",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"

)

# ============ Funciones Helper ============

def check_api_health() -> bool:
    """Verifica que la API esté funcionando"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def send_message(message: str, conversation_id: Optional[str] = None) -> Dict[str, Any]:
    """Envía mensaje a la API"""
    try:
        payload = {"message": message}
        if conversation_id:
            payload["conversation_id"] = conversation_id

        response = requests.post(
            f"{API_URL}/chat",
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            return response.json()
        else:
            return {
                "error": f"Error {response.status_code}: {response.text}",
                "response": "Error al procesar la solicitud"
            }
    except requests.exceptions.Timeout:
        return {
            "error": "Timeout: La consulta está tardando demasiado",
            "response": "La consulta está tardando más de lo esperado. Por favor intenta de nuevo."
        }
    except Exception as e:
        return {
            "error": str(e),
            "response": f"Error de conexión: {str(e)}"
        }


def render_chart(chart_config: Dict[str, Any]):
    """Renderiza gráfico de Plotly"""
    try:
        config = chart_config.get("config", {})

        # Crear figura de Plotly desde el JSON
        fig = go.Figure(config)

        # Mostrar en Streamlit
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error renderizando gráfico: {e}")


def render_table(results: list):
    """Renderiza tabla de resultados"""
    if results:
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)


def render_kpis(kpis: Dict[str, Any]):
    """Renderiza KPIs en columnas"""
    if not kpis:
        return

    # Crear columnas dinámicamente
    num_kpis = len(kpis)
    cols = st.columns(min(num_kpis, 4))  # Máximo 4 columnas

    for idx, (key, kpi_data) in enumerate(kpis.items()):
        col_idx = idx % 4
        with cols[col_idx]:
            st.metric(
                label=kpi_data.get("name", key),
                value=kpi_data.get("formatted_value", "N/A"),
                help=kpi_data.get("description", "")
            )


# ============ Sidebar ============

with st.sidebar:
    st.title("⚙️ Configuración")

    # Health check
    if check_api_health():
        st.success("✅ API conectada")
    else:
        st.error("❌ API no disponible")
        st.info(f"Verifica que la API esté corriendo en {API_URL}")

    st.divider()

    # Opciones
    show_raw_response = st.checkbox("Mostrar respuesta raw (JSON)", value=False)
    show_sql = st.checkbox("Mostrar SQL generado", value=True)

    st.divider()

    # Ejemplos de queries
    st.subheader("📝 Ejemplos")

    examples = [
        "¿Cuántas ventas preventivas hay?",
        "Muéstrame los 10 productos más vendidos",
        "Calcula el revenue total",
        "Gráfica de ventas por mes",
        "Compara preventivas vs correctivas",
        "¿Cuál es el ticket promedio?",
    ]

    for example in examples:
        if st.button(example, key=f"example_{example}", use_container_width=True):
            st.session_state.example_query = example

    st.divider()

    # Botón para limpiar caché
    if st.button("🗑️ Limpiar caché de la API", use_container_width=True):
        try:
            response = requests.delete(f"{API_URL}/cache")
            if response.status_code == 200:
                st.success("✅ Caché limpiado")
            else:
                st.error("❌ Error limpiando caché")
        except:
            st.error("❌ No se pudo conectar con la API")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Analytics")

    if st.sidebar.button("Ver Métricas"):
        try:
            metrics_response = requests.get(f"{API_URL}/metrics?days=7", timeout=10)
            if metrics_response.status_code == 200:
                metrics = metrics_response.json()

                st.markdown("## 📈 Métricas de los Últimos 7 Días")

                # Métricas generales
                gen = metrics['general']
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Interacciones", gen['total_interactions'])
                col2.metric("Rating Promedio", f"{gen['avg_rating']}/5.0")
                col3.metric("Tiempo Respuesta (avg)", f"{gen['avg_response_time_ms']}ms")

                # Distribución de ratings
                if metrics['rating_distribution']:
                    st.markdown("### Distribución de Ratings")
                    rating_df = pd.DataFrame([
                        {'Rating': f"{k}⭐", 'Cantidad': v}
                        for k, v in metrics['rating_distribution'].items()
                    ])
                    st.bar_chart(rating_df.set_index('Rating'))

                # Charts más usados
                if metrics['top_charts']:
                    st.markdown("### Gráficos Más Usados")
                    charts_df = pd.DataFrame(metrics['top_charts'])
                    st.dataframe(charts_df)

                # Errores comunes
                if metrics['top_errors']:
                    st.markdown("### Errores Más Comunes")
                    errors_df = pd.DataFrame(metrics['top_errors'])
                    st.dataframe(errors_df)
        except Exception as e:
            st.error(f"Error obteniendo métricas: {e}")
# ============ Main App ============

st.title("📊 Chatbot Analítico")
st.markdown("Haz preguntas sobre tus datos de ventas en **lenguaje natural**")

# Inicializar historial de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Renderizar componentes adicionales si existen
        if "chart_config" in message and message["chart_config"]:
            render_chart(message["chart_config"])

        if "results" in message and message["results"]:
            with st.expander("📋 Ver datos completos"):
                render_table(message["results"])

        if "kpis" in message and message["kpis"]:
            st.markdown("**KPIs:**")
            render_kpis(message["kpis"])

        if "sql_query" in message and message["sql_query"] and show_sql:
            with st.expander("🔍 SQL generado"):
                st.code(message["sql_query"], language="sql")

# Input del usuario
user_input = st.chat_input("Escribe tu pregunta aquí...")

# Si hay ejemplo seleccionado, usarlo
if "example_query" in st.session_state:
    user_input = st.session_state.example_query
    del st.session_state.example_query

# Procesar input
if user_input:
    # Agregar mensaje del usuario
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Mostrar mensaje del usuario
    with st.chat_message("user"):
        st.markdown(user_input)

    # Mostrar spinner mientras procesa
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            # Llamar a la API
            response = send_message(user_input)

            # Mostrar respuesta
            st.markdown(response.get("response", "No hubo respuesta"))

            # Renderizar gráfico si existe
            if response.get("chart_config"):
                render_chart(response["chart_config"])
                
            if 'feedback_id' in response:
                st.markdown("---")
                st.markdown("**¿Qué te pareció esta respuesta?**")

                col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 3])

                rating = None
                with col1:
                    if st.button("⭐", key=f"rate1_{response['feedback_id']}"):
                        rating = 1
                with col2:
                    if st.button("⭐⭐", key=f"rate2_{response['feedback_id']}"):
                        rating = 2
                with col3:
                    if st.button("⭐⭐⭐", key=f"rate3_{response['feedback_id']}"):
                        rating = 3
                with col4:
                    if st.button("⭐⭐⭐⭐", key=f"rate4_{response['feedback_id']}"):
                        rating = 4
                with col5:
                    if st.button("⭐⭐⭐⭐⭐", key=f"rate5_{response['feedback_id']}"):
                        rating = 5

                if rating:
                    # Opcional: pedir comentario para ratings bajos
                    feedback_text = None
                    if rating <= 3:
                        feedback_text = st.text_input(
                            "¿Qué podemos mejorar?",
                            key=f"feedback_text_{response['feedback_id']}"
                        )

                    # Enviar feedback
                    try:
                        feedback_response = requests.post(
                            f"{API_URL}/feedback",
                            json={
                                "feedback_id": response['feedback_id'],
                                "rating": rating,
                                "feedback_text": feedback_text
                            },
                            timeout=5
                        )
                        if feedback_response.status_code == 200:
                            st.success(f"¡Gracias por tu valoración de {rating} estrellas!")
                    except Exception as e:
                        st.error(f"Error enviando feedback: {e}")

            # Renderizar tabla si existen resultados
            if response.get("results"):
                with st.expander("📋 Ver datos completos"):
                    render_table(response["results"])

            # Renderizar KPIs si existen
            if response.get("kpis"):
                st.markdown("**KPIs:**")
                render_kpis(response["kpis"])

            # Mostrar SQL si existe
            if response.get("sql_query") and show_sql:
                with st.expander("🔍 SQL generado"):
                    st.code(response["sql_query"], language="sql")

            # Mostrar respuesta raw si está activado
            if show_raw_response:
                with st.expander("📄 Respuesta raw (JSON)"):
                    st.json(response)

            # Mostrar error si existe
            if response.get("error"):
                st.error(f"⚠️ Error: {response['error']}")

            # Agregar respuesta al historial
            st.session_state.messages.append({
                "role": "assistant",
                "content": response.get("response", "No hubo respuesta"),
                "chart_config": response.get("chart_config"),
                "results": response.get("results"),
                "kpis": response.get("kpis"),
                "sql_query": response.get("sql_query"),
                "intent": response.get("intent")
            })

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8em;'>
    💡 Tip: Puedes pedir gráficas, KPIs, consultas SQL o hacer preguntas generales<br>
    Ejemplos: "Gráfica de ventas por producto" • "Calcula el revenue total" • "¿Cuántas ventas correctivas hay?"
</div>
""", unsafe_allow_html=True)
