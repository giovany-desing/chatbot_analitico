"""
Herramienta para generar visualizaciones con Plotly.
Retorna configuración JSON que puede ser renderizada en frontend.
"""

from langchain_core.tools import BaseTool
from typing import List, Dict, Any, Optional, Literal
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import logging
import json

logger = logging.getLogger(__name__)


ChartType = Literal["bar", "line", "pie", "scatter", "histogram", "box", "heatmap"]


class VizTool(BaseTool):
    """
    Herramienta para generar gráficas con Plotly.
    
    Input: Dict con datos y configuración
    Output: JSON de configuración de Plotly
    """

    name: str = "visualization"
    description: str = """
    Genera visualizaciones de datos usando Plotly.
    
    Input: Debe ser un dict con:
    - data: Lista de dicts con los datos
    - chart_type: Tipo de gráfica (bar, line, pie, scatter, histogram)
    - x_column: Columna para eje X
    - y_column: Columna para eje Y (opcional para algunos tipos)
    - title: Título de la gráfica
    
    Output: Configuración JSON de Plotly
    """

    def _run(
        self,
        data: List[Dict[str, Any]],
        chart_type: ChartType = "bar",
        x_column: Optional[str] = None,
        y_column: Optional[str] = None,
        title: Optional[str] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Genera la gráfica.
        
        Args:
            data: Lista de dicts con los datos
            chart_type: Tipo de gráfica
            x_column: Columna para eje X
            y_column: Columna para eje Y
            title: Título
            x_label: Etiqueta eje X
            y_label: Etiqueta eje Y
        
        Returns:
            Dict con configuración de Plotly
        """
        try:
            # Convertir a DataFrame
            df = pd.DataFrame(data)

            logger.info(f"Creating {chart_type} chart with {len(df)} rows")

            # Auto-detectar columnas si no se especifican
            if not x_column and len(df.columns) > 0:
                x_column = df.columns[0]

            if not y_column and len(df.columns) > 1:
                # Buscar primera columna numérica
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    y_column = numeric_cols[0]
                else:
                    y_column = df.columns[1]

            # Generar título automático si no existe
            if not title:
                title = f"{chart_type.capitalize()} de {y_column or 'datos'}"

            # Crear figura según tipo
            if chart_type == "bar":
                fig = px.bar(
                    df,
                    x=x_column,
                    y=y_column,
                    title=title,
                    labels={x_column: x_label or x_column, y_column: y_label or y_column}
                )

            elif chart_type == "line":
                fig = px.line(
                    df,
                    x=x_column,
                    y=y_column,
                    title=title,
                    labels={x_column: x_label or x_column, y_column: y_label or y_column},
                    markers=True
                )

            elif chart_type == "pie":
                fig = px.pie(
                    df,
                    names=x_column,
                    values=y_column,
                    title=title
                )

            elif chart_type == "scatter":
                fig = px.scatter(
                    df,
                    x=x_column,
                    y=y_column,
                    title=title,
                    labels={x_column: x_label or x_column, y_column: y_label or y_column}
                )

            elif chart_type == "histogram":
                fig = px.histogram(
                    df,
                    x=x_column,
                    title=title,
                    labels={x_column: x_label or x_column}
                )

            elif chart_type == "box":
                fig = px.box(
                    df,
                    y=y_column,
                    title=title,
                    labels={y_column: y_label or y_column}
                )

            elif chart_type == "heatmap":
                # Crear matriz de correlación si hay múltiples columnas numéricas
                numeric_df = df.select_dtypes(include=['number'])
                if len(numeric_df.columns) > 1:
                    corr = numeric_df.corr()
                    fig = px.imshow(
                        corr,
                        title=title or "Matriz de Correlación",
                        labels=dict(color="Correlación"),
                        x=corr.columns,
                        y=corr.columns,
                        color_continuous_scale="RdBu"
                    )
                else:
                    raise ValueError("Heatmap requiere múltiples columnas numéricas")

            else:
                raise ValueError(f"Unsupported chart type: {chart_type}")

            # Mejorar diseño
            fig.update_layout(
                template="plotly_white",
                hovermode="closest",
                showlegend=True if chart_type in ["line", "scatter"] else False
            )

            # Convertir a JSON
            chart_json = json.loads(fig.to_json())

            logger.info(f"✓ Chart generated successfully")

            return {
                "chart_type": chart_type,
                "config": chart_json,
                "data_points": len(df),
                "columns": list(df.columns)
            }

        except Exception as e:
            logger.error(f"Error generating chart: {e}")
            raise

    async def _arun(self, *args, **kwargs):
        """Versión async (usa sync)"""
        return self._run(*args, **kwargs)

    def auto_select_chart_type(self, df: pd.DataFrame) -> ChartType:
        """
        Selecciona automáticamente el mejor tipo de gráfica.
        
        Args:
            df: DataFrame con los datos
        
        Returns:
            Tipo de gráfica recomendado
        """
        num_rows = len(df)
        num_numeric = len(df.select_dtypes(include=['number']).columns)
        num_categorical = len(df.select_dtypes(include=['object', 'category']).columns)

        # Reglas heurísticas
        if num_categorical == 1 and num_numeric == 1:
            if num_rows <= 10:
                return "bar"  # Pocas categorías → barras
            else:
                return "line"  # Muchas categorías (probablemente tiempo) → línea

        elif num_categorical == 0 and num_numeric == 1:
            return "histogram"  # Solo numéricos → histograma

        elif num_categorical == 0 and num_numeric == 2:
            return "scatter"  # 2 numéricos → scatter

        elif num_categorical == 1 and num_numeric == 1 and num_rows <= 7:
            return "pie"  # Pocas categorías con un valor → pie

        elif num_numeric > 2:
            return "heatmap"  # Muchos numéricos → correlación

        else:
            return "bar"  # Default


# Instancia global
viz_tool = VizTool()


# Para testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== Testing VizTool ===\n")

    # Test 1: Bar chart
    print("1. Bar chart:")
    data = [
        {"producto": "Laptop", "ventas": 120},
        {"producto": "Mouse", "ventas": 450},
        {"producto": "Teclado", "ventas": 230}
    ]

    result = viz_tool._run(
        data=data,
        chart_type="bar",
        x_column="producto",
        y_column="ventas",
        title="Ventas por Producto"
    )

    print(f"   Chart type: {result['chart_type']}")
    print(f"   Data points: {result['data_points']}\n")

    # Test 2: Line chart
    print("2. Line chart:")
    data = [
        {"mes": "Enero", "revenue": 15000},
        {"mes": "Febrero", "revenue": 18000},
        {"mes": "Marzo", "revenue": 22000}
    ]

    result = viz_tool._run(
        data=data,
        chart_type="line",
        x_column="mes",
        y_column="revenue",
        title="Revenue por Mes"
    )

    print(f"   Chart type: {result['chart_type']}")
    print(f"   Data points: {result['data_points']}\n")

    # Test 3: Auto-select
    print("3. Auto-select chart type:")
    df = pd.DataFrame(data)
    recommended = viz_tool.auto_select_chart_type(df)
    print(f"   Recommended: {recommended}")