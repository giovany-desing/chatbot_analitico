import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class ProfessionalVizTool:
    """
    Generador de gráficas de nivel empresarial.
    """

    # Paletas de colores profesionales
    PALETTES = {
        'corporate': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E'],
        'modern': ['#264653', '#2A9D8F', '#E9C46A', '#F4A261', '#E76F51'],
        'elegant': ['#006BA6', '#0496FF', '#FFBC42', '#D81159', '#8F2D56'],
        'professional': ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087'],
    }

    # Configuración de fuentes
    FONT_CONFIG = {
        'family': 'Arial, Helvetica, sans-serif',
        'title_size': 24,
        'axis_size': 14,
        'tick_size': 12,
        'legend_size': 12
    }

    def create_chart(
        self,
        data: List[Dict],
        chart_type: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Crea gráfica profesional.

        Args:
            data: Lista de diccionarios con datos
            chart_type: Tipo de gráfica
            config: Configuración (x_column, y_column, title, etc.)

        Returns:
            Dict con configuración JSON de Plotly
        """
        df = pd.DataFrame(data)

        logger.info(f"Creating professional {chart_type} chart")

        # Crear figura según tipo
        if chart_type == 'bar':
            fig = self._create_bar_chart(df, config)
        elif chart_type == 'line':
            fig = self._create_line_chart(df, config)
        elif chart_type == 'pie':
            fig = self._create_pie_chart(df, config)
        elif chart_type == 'scatter':
            fig = self._create_scatter_chart(df, config)
        elif chart_type == 'histogram':
            fig = self._create_histogram(df, config)
        else:
            fig = self._create_bar_chart(df, config)  # Fallback

        # Aplicar tema profesional
        fig = self._apply_professional_theme(fig)

        # Convertir a JSON
        import json
        chart_json = json.loads(fig.to_json())

        return {
            'chart_type': chart_type,
            'config': chart_json,
            'data_points': len(df),
            'columns': list(df.columns)
        }

    def _create_bar_chart(self, df: pd.DataFrame, config: Dict) -> go.Figure:
        """Gráfica de barras profesional con gradiente y valores"""
        x_col = config.get('x_column')
        y_col = config.get('y_column')

        # Ordenar si tiene sentido
        if df[y_col].dtype in ['int64', 'float64']:
            df_sorted = df.sort_values(y_col, ascending=False)
        else:
            df_sorted = df

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=df_sorted[x_col],
            y=df_sorted[y_col],
            marker=dict(
                color=df_sorted[y_col],
                colorscale='Blues',
                showscale=False,
                line=dict(color='white', width=1)
            ),
            text=df_sorted[y_col],
            texttemplate='<b>%{text:,.0f}</b>',
            textposition='outside',
            textfont=dict(size=self.FONT_CONFIG['tick_size']),
            hovertemplate='<b>%{x}</b><br>' +
                         f'{y_col}: <b>%{{y:,.2f}}</b><br>' +
                         '<extra></extra>',
            name=y_col
        ))

        fig.update_layout(
            title=dict(
                text=config.get('title', f'{y_col} por {x_col}'),
                font=dict(
                    size=self.FONT_CONFIG['title_size'],
                    color='#1a1a1a',
                    family=self.FONT_CONFIG['family']
                ),
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                title=dict(
                    text=config.get('x_label', x_col),
                    font=dict(size=self.FONT_CONFIG['axis_size'])
                ),
                tickangle=-45 if len(df_sorted) > 10 else 0,
                showgrid=False,
                showline=True,
                linecolor='#e0e0e0'
            ),
            yaxis=dict(
                title=dict(
                    text=config.get('y_label', y_col),
                    font=dict(size=self.FONT_CONFIG['axis_size'])
                ),
                showgrid=True,
                gridcolor='#f0f0f0',
                showline=True,
                linecolor='#e0e0e0'
            ),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        return fig

    def _create_line_chart(self, df: pd.DataFrame, config: Dict) -> go.Figure:
        """Gráfica de línea profesional con área sombreada y tendencia"""
        x_col = config.get('x_column')
        y_col = config.get('y_column')

        fig = go.Figure()

        # Línea principal con área
        fig.add_trace(go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode='lines+markers',
            name=y_col,
            line=dict(
                color='#2E86AB',
                width=3
            ),
            marker=dict(
                size=8,
                color='#2E86AB',
                line=dict(width=2, color='white')
            ),
            fill='tozeroy',
            fillcolor='rgba(46, 134, 171, 0.1)',
            hovertemplate='<b>%{x}</b><br>' +
                         f'{y_col}: <b>%{{y:,.2f}}</b><br>' +
                         '<extra></extra>'
        ))

        # Línea de tendencia
        if len(df) >= 3:
            from scipy import stats
            x_numeric = np.arange(len(df))
            slope, intercept, r_value, _, _ = stats.linregress(x_numeric, df[y_col])
            trend_line = slope * x_numeric + intercept

            fig.add_trace(go.Scatter(
                x=df[x_col],
                y=trend_line,
                mode='lines',
                name='Tendencia',
                line=dict(
                    color='#F18F01',
                    width=2,
                    dash='dash'
                ),
                hovertemplate='Tendencia: <b>%{y:,.2f}</b><extra></extra>'
            ))

        fig.update_layout(
            title=dict(
                text=config.get('title', f'Evolución de {y_col}'),
                font=dict(
                    size=self.FONT_CONFIG['title_size'],
                    color='#1a1a1a',
                    family=self.FONT_CONFIG['family']
                ),
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                title=dict(
                    text=config.get('x_label', x_col),
                    font=dict(size=self.FONT_CONFIG['axis_size'])
                ),
                showgrid=False,
                showline=True,
                linecolor='#e0e0e0'
            ),
            yaxis=dict(
                title=dict(
                    text=config.get('y_label', y_col),
                    font=dict(size=self.FONT_CONFIG['axis_size'])
                ),
                showgrid=True,
                gridcolor='#f0f0f0',
                showline=True,
                linecolor='#e0e0e0',
                zeroline=True,
                zerolinecolor='#cccccc'
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1,
                font=dict(size=self.FONT_CONFIG['legend_size'])
            ),
            hovermode='x unified'
        )

        return fig

    def _create_pie_chart(self, df: pd.DataFrame, config: Dict) -> go.Figure:
        """Gráfica de pastel profesional tipo donut con pull-out"""
        x_col = config.get('x_column')
        y_col = config.get('y_column')

        # Ordenar por valor
        df_sorted = df.sort_values(y_col, ascending=False)

        # Pull para el valor más grande
        pull = [0.1 if i == 0 else 0 for i in range(len(df_sorted))]

        # Colores
        colors = self.PALETTES['modern'][:len(df_sorted)]

        fig = go.Figure(data=[go.Pie(
            labels=df_sorted[x_col],
            values=df_sorted[y_col],
            pull=pull,
            hole=0.4,  # Donut style
            marker=dict(
                colors=colors,
                line=dict(color='white', width=3)
            ),
            textinfo='label+percent',
            textposition='outside',
            textfont=dict(
                size=self.FONT_CONFIG['tick_size'],
                color='#1a1a1a'
            ),
            hovertemplate='<b>%{label}</b><br>' +
                         'Valor: <b>%{value:,.2f}</b><br>' +
                         'Porcentaje: <b>%{percent}</b><br>' +
                         '<extra></extra>'
        )])

        # Agregar texto central
        total = df_sorted[y_col].sum()
        fig.add_annotation(
            text=f'<b>Total</b><br>{total:,.0f}',
            x=0.5, y=0.5,
            font=dict(size=16, color='#1a1a1a'),
            showarrow=False
        )

        fig.update_layout(
            title=dict(
                text=config.get('title', f'Distribución de {y_col}'),
                font=dict(
                    size=self.FONT_CONFIG['title_size'],
                    color='#1a1a1a',
                    family=self.FONT_CONFIG['family']
                ),
                x=0.5,
                xanchor='center'
            ),
            paper_bgcolor='white',
            showlegend=True,
            legend=dict(
                orientation='v',
                yanchor='middle',
                y=0.5,
                xanchor='left',
                x=1.05,
                font=dict(size=self.FONT_CONFIG['legend_size'])
            )
        )

        return fig

    def _create_scatter_chart(self, df: pd.DataFrame, config: Dict) -> go.Figure:
        """Gráfica de dispersión profesional con línea de tendencia"""
        x_col = config.get('x_column')
        y_col = config.get('y_column')

        fig = go.Figure()

        # Scatter plot
        fig.add_trace(go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode='markers',
            name='Datos',
            marker=dict(
                size=10,
                color=df[y_col],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=y_col),
                line=dict(width=1, color='white')
            ),
            hovertemplate=f'<b>{x_col}: %{{x:,.2f}}</b><br>' +
                         f'{y_col}: <b>%{{y:,.2f}}</b><br>' +
                         '<extra></extra>'
        ))

        # Línea de regresión
        if len(df) >= 3:
            from scipy import stats
            slope, intercept, r_value, _, _ = stats.linregress(df[x_col], df[y_col])
            line_x = np.array([df[x_col].min(), df[x_col].max()])
            line_y = slope * line_x + intercept

            fig.add_trace(go.Scatter(
                x=line_x,
                y=line_y,
                mode='lines',
                name=f'Regresión (R²={r_value**2:.3f})',
                line=dict(color='#F18F01', width=2, dash='dash'),
                hovertemplate='<extra></extra>'
            ))

        fig.update_layout(
            title=dict(
                text=config.get('title', f'{y_col} vs {x_col}'),
                font=dict(
                    size=self.FONT_CONFIG['title_size'],
                    color='#1a1a1a',
                    family=self.FONT_CONFIG['family']
                ),
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                title=dict(
                    text=config.get('x_label', x_col),
                    font=dict(size=self.FONT_CONFIG['axis_size'])
                ),
                showgrid=True,
                gridcolor='#f0f0f0'
            ),
            yaxis=dict(
                title=dict(
                    text=config.get('y_label', y_col),
                    font=dict(size=self.FONT_CONFIG['axis_size'])
                ),
                showgrid=True,
                gridcolor='#f0f0f0'
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=True
        )

        return fig

    def _create_histogram(self, df: pd.DataFrame, config: Dict) -> go.Figure:
        """Histograma profesional con KDE overlay"""
        x_col = config.get('x_column')

        fig = go.Figure()

        # Histograma
        fig.add_trace(go.Histogram(
            x=df[x_col],
            nbinsx=30,
            name='Frecuencia',
            marker=dict(
                color='#2E86AB',
                line=dict(color='white', width=1)
            ),
            opacity=0.7,
            hovertemplate='Rango: %{x}<br>Frecuencia: <b>%{y}</b><extra></extra>'
        ))

        fig.update_layout(
            title=dict(
                text=config.get('title', f'Distribución de {x_col}'),
                font=dict(
                    size=self.FONT_CONFIG['title_size'],
                    color='#1a1a1a',
                    family=self.FONT_CONFIG['family']
                ),
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                title=dict(
                    text=config.get('x_label', x_col),
                    font=dict(size=self.FONT_CONFIG['axis_size'])
                ),
                showgrid=False
            ),
            yaxis=dict(
                title=dict(
                    text='Frecuencia',
                    font=dict(size=self.FONT_CONFIG['axis_size'])
                ),
                showgrid=True,
                gridcolor='#f0f0f0'
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            bargap=0.1
        )

        return fig

    def _apply_professional_theme(self, fig: go.Figure) -> go.Figure:
        """Aplica configuración profesional global"""
        fig.update_layout(
            # Márgenes
            margin=dict(l=80, r=80, t=100, b=80),

            # Fuentes
            font=dict(
                family=self.FONT_CONFIG['family'],
                size=self.FONT_CONFIG['tick_size'],
                color='#1a1a1a'
            ),

            # Hover mejorado
            hoverlabel=dict(
                bgcolor='white',
                bordercolor='#1a1a1a',
                font=dict(
                    size=self.FONT_CONFIG['tick_size'],
                    family=self.FONT_CONFIG['family'],
                    color='#1a1a1a'
                )
            ),

            # Animaciones suaves
            transition=dict(
                duration=500,
                easing='cubic-in-out'
            ),

            # Toolbar personalizado
            modebar=dict(
                bgcolor='rgba(255,255,255,0.8)',
                color='#1a1a1a',
                activecolor='#2E86AB',
                orientation='v'
            )
        )

        return fig


# Instancia global
professional_viz_tool = ProfessionalVizTool()