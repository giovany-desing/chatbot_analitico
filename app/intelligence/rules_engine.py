import pandas as pd
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ConfidenceLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class RuleResult:
    chart_type: Optional[str]
    confidence: ConfidenceLevel
    config: Optional[Dict[str, Any]]
    reasoning: str


class DeterministicRulesEngine:
    """
    Motor de reglas determinísticas para decisiones rápidas.
    Cubre ~70% de casos comunes sin usar IA.
    """

    def apply(self, query: str, results: List[Dict]) -> RuleResult:
        """
        Aplica reglas determinísticas.

        Returns:
            RuleResult con chart_type, confidence y config
        """
        if not results:
            return RuleResult(None, ConfidenceLevel.LOW, None, "Sin datos")

        df = pd.DataFrame(results)
        query_lower = query.lower()

        # REGLA 1: Keywords explícitas de tipo de gráfica
        explicit_result = self._check_explicit_keywords(query_lower, df)
        if explicit_result:
            return explicit_result

        # REGLA 2: Detectar columnas de fecha (temporal)
        date_result = self._check_date_columns(df, query_lower)
        if date_result:
            return date_result

        # REGLA 3: Top N / Ranking
        topn_result = self._check_top_n(query_lower, df)
        if topn_result:
            return topn_result

        # REGLA 4: Pocas categorías (pie chart)
        pie_result = self._check_pie_chart(query_lower, df)
        if pie_result:
            return pie_result

        # REGLA 5: Comparación explícita
        comparison_result = self._check_comparison(query_lower, df)
        if comparison_result:
            return comparison_result

        # REGLA 6: Distribución
        distribution_result = self._check_distribution(query_lower, df)
        if distribution_result:
            return distribution_result

        # REGLA 7: Scatter (correlación)
        scatter_result = self._check_scatter(query_lower, df)
        if scatter_result:
            return scatter_result

        # Sin match
        return RuleResult(None, ConfidenceLevel.LOW, None, "No rule matched")

    def _check_explicit_keywords(self, query: str, df: pd.DataFrame) -> Optional[RuleResult]:
        """Detecta keywords explícitas del tipo de gráfica"""
        keywords_map = {
            'line': ['línea', 'line', 'tendencia', 'evolución', 'temporal'],
            'bar': ['barra', 'barras', 'bar'],
            'pie': ['pastel', 'pie', 'torta', 'circular'],
            'scatter': ['dispersión', 'scatter', 'puntos'],
            'histogram': ['histogram', 'histograma', 'frecuencia']  # Removido 'distribución' - puede ser pie chart
        }

        for chart_type, keywords in keywords_map.items():
            if any(kw in query for kw in keywords):
                config = self._auto_config(chart_type, df)
                return RuleResult(
                    chart_type=chart_type,
                    confidence=ConfidenceLevel.HIGH,
                    config=config,
                    reasoning=f"Keyword explícito detectado: {chart_type}"
                )
        return None

    def _check_date_columns(self, df: pd.DataFrame, query: str) -> Optional[RuleResult]:
        """Detecta columnas de fecha para line chart"""
        date_keywords = ['fecha', 'date', 'mes', 'month', 'año', 'year', 'trimestre', 'quarter']

        for col in df.columns:
            if any(kw in col.lower() for kw in date_keywords):
                config = self._line_chart_config(df, col)
                return RuleResult(
                    chart_type='line',
                    confidence=ConfidenceLevel.HIGH,
                    config=config,
                    reasoning=f"Columna temporal detectada: {col}"
                )
        return None

    def _check_top_n(self, query: str, df: pd.DataFrame) -> Optional[RuleResult]:
        """Detecta queries de tipo Top N"""
        top_keywords = ['top', 'mejores', 'más vendidos', 'ranking', 'primeros', 'mayores']

        if any(kw in query for kw in top_keywords) and len(df) <= 20:
            config = self._bar_chart_config(df)
            return RuleResult(
                chart_type='bar',
                confidence=ConfidenceLevel.HIGH,
                config=config,
                reasoning="Top N detectado con cantidad razonable de items"
            )
        return None

    def _check_pie_chart(self, query: str, df: pd.DataFrame) -> Optional[RuleResult]:
        """Detecta casos apropiados para pie chart"""
        pie_keywords = ['distribución', 'proporción', 'porcentaje', 'reparto']

        if (any(kw in query for kw in pie_keywords) and 2 <= len(df) <= 7):
            config = self._pie_chart_config(df)
            return RuleResult(
                chart_type='pie',
                confidence=ConfidenceLevel.HIGH,
                config=config,
                reasoning="Proporción con pocas categorías (2-7)"
            )
        return None

    def _check_comparison(self, query: str, df: pd.DataFrame) -> Optional[RuleResult]:
        """Detecta comparaciones explícitas"""
        comparison_keywords = ['vs', 'versus', 'compara', 'comparación', 'diferencia']

        if any(kw in query for kw in comparison_keywords):
            config = self._bar_chart_config(df)
            return RuleResult(
                chart_type='bar',
                confidence=ConfidenceLevel.HIGH,
                config=config,
                reasoning="Comparación explícita detectada"
            )
        return None

    def _check_distribution(self, query: str, df: pd.DataFrame) -> Optional[RuleResult]:
        """Detecta análisis de distribución"""
        if 'histogram' in query or 'frecuencia' in query:
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                config = self._histogram_config(df, numeric_cols[0])
                return RuleResult(
                    chart_type='histogram',
                    confidence=ConfidenceLevel.HIGH,
                    config=config,
                    reasoning="Distribución solicitada"
                )
        return None

    def _check_scatter(self, query: str, df: pd.DataFrame) -> Optional[RuleResult]:
        """Detecta necesidad de scatter plot"""
        scatter_keywords = ['correlación', 'relación', 'scatter']
        numeric_cols = df.select_dtypes(include=['number']).columns

        if any(kw in query for kw in scatter_keywords) and len(numeric_cols) >= 2:
            config = self._scatter_config(df, numeric_cols[0], numeric_cols[1])
            return RuleResult(
                chart_type='scatter',
                confidence=ConfidenceLevel.HIGH,
                config=config,
                reasoning="Correlación entre dos variables numéricas"
            )
        return None

    def _auto_config(self, chart_type: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Genera configuración automática según tipo"""
        if chart_type == 'line':
            return self._line_chart_config(df, df.columns[0])
        elif chart_type == 'bar':
            return self._bar_chart_config(df)
        elif chart_type == 'pie':
            return self._pie_chart_config(df)
        elif chart_type == 'scatter':
            numeric_cols = df.select_dtypes(include=['number']).columns
            return self._scatter_config(df, numeric_cols[0], numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0])
        elif chart_type == 'histogram':
            numeric_cols = df.select_dtypes(include=['number']).columns
            return self._histogram_config(df, numeric_cols[0])
        return {}

    def _line_chart_config(self, df: pd.DataFrame, x_col: str) -> Dict[str, Any]:
        numeric_cols = df.select_dtypes(include=['number']).columns
        y_col = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[1]
        return {
            'chart_type': 'line',
            'x_column': x_col,
            'y_column': y_col,
            'title': f'Evolución de {y_col}',
            'x_label': x_col,
            'y_label': y_col
        }

    def _bar_chart_config(self, df: pd.DataFrame) -> Dict[str, Any]:
        x_col = df.columns[0]
        numeric_cols = df.select_dtypes(include=['number']).columns
        y_col = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[1]
        return {
            'chart_type': 'bar',
            'x_column': x_col,
            'y_column': y_col,
            'title': f'{y_col} por {x_col}',
            'x_label': x_col,
            'y_label': y_col
        }

    def _pie_chart_config(self, df: pd.DataFrame) -> Dict[str, Any]:
        x_col = df.columns[0]
        numeric_cols = df.select_dtypes(include=['number']).columns
        y_col = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[1]
        return {
            'chart_type': 'pie',
            'x_column': x_col,
            'y_column': y_col,
            'title': f'Distribución de {y_col}',
            'x_label': x_col,
            'y_label': y_col
        }

    def _histogram_config(self, df: pd.DataFrame, col: str) -> Dict[str, Any]:
        return {
            'chart_type': 'histogram',
            'x_column': col,
            'y_column': None,
            'title': f'Distribución de {col}',
            'x_label': col,
            'y_label': 'Frecuencia'
        }

    def _scatter_config(self, df: pd.DataFrame, x_col: str, y_col: str) -> Dict[str, Any]:
        return {
            'chart_type': 'scatter',
            'x_column': x_col,
            'y_column': y_col,
            'title': f'{y_col} vs {x_col}',
            'x_label': x_col,
            'y_label': y_col
        }