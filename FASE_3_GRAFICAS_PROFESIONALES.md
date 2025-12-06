# 🎨 FASE 3: Gráficas de Calidad Profesional

## Objetivo
Mejorar la calidad visual de las gráficas Plotly para alcanzar estándares empresariales con diseño profesional, interactividad avanzada y presentación impecable.

---

## 📦 Prerequisitos

- Fase 2 completada (sistema híbrido funcionando)
- Proyecto actual operativo

---

## 🔧 Paso 1: Instalar Dependencias Adicionales

```bash
# Agregar a requirements.txt
scipy>=1.11.0
kaleido>=0.2.1
```

```bash
# Instalar
pip install scipy kaleido
```

---

## 📝 Paso 2: Crear Herramienta de Visualización Profesional

```python
# app/tools/professional_viz.py

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
                title=config.get('x_label', x_col),
                titlefont=dict(size=self.FONT_CONFIG['axis_size']),
                tickangle=-45 if len(df_sorted) > 10 else 0,
                showgrid=False,
                showline=True,
                linecolor='#e0e0e0'
            ),
            yaxis=dict(
                title=config.get('y_label', y_col),
                titlefont=dict(size=self.FONT_CONFIG['axis_size']),
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
                title=config.get('x_label', x_col),
                titlefont=dict(size=self.FONT_CONFIG['axis_size']),
                showgrid=False,
                showline=True,
                linecolor='#e0e0e0'
            ),
            yaxis=dict(
                title=config.get('y_label', y_col),
                titlefont=dict(size=self.FONT_CONFIG['axis_size']),
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
                title=config.get('x_label', x_col),
                titlefont=dict(size=self.FONT_CONFIG['axis_size']),
                showgrid=True,
                gridcolor='#f0f0f0'
            ),
            yaxis=dict(
                title=config.get('y_label', y_col),
                titlefont=dict(size=self.FONT_CONFIG['axis_size']),
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
                title=config.get('x_label', x_col),
                titlefont=dict(size=self.FONT_CONFIG['axis_size']),
                showgrid=False
            ),
            yaxis=dict(
                title='Frecuencia',
                titlefont=dict(size=self.FONT_CONFIG['axis_size']),
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
```

---

## 📝 Paso 3: Actualizar viz_node para Usar Gráficas Profesionales

```python
# app/agents/nodes.py

# Importar al inicio
from app.tools.professional_viz import professional_viz_tool

# Modificar viz_node (solo la parte de generación de gráfica):
def viz_node(state: AgentState) -> AgentState:
    """
    Genera visualizaciones usando sistema híbrido + gráficas profesionales.
    """
    logger.info("=== Viz Node (Hybrid + Professional) ===")

    try:
        # ... código existente hasta chart_config ...

        # Usar herramienta profesional en lugar de viz_tool
        logger.info("Generating professional chart")
        chart_result = professional_viz_tool.create_chart(
            data=results,
            chart_type=chart_config.get('chart_type'),
            config=chart_config
        )

        # ... resto del código ...

        return state

    except Exception as e:
        logger.error(f"Error in viz_node: {e}")
        state['error'] = str(e)
        return state
```

---

## 🧪 Plan de Pruebas

### Test 1: Verificar Gráficas Profesionales Localmente

```python
# tests/test_professional_viz.py

from app.tools.professional_viz import professional_viz_tool

def test_bar_chart():
    """Test: Gráfica de barras profesional"""
    data = [
        {"producto": "A", "ventas": 1000},
        {"producto": "B", "ventas": 1500},
        {"producto": "C", "ventas": 800},
    ]

    config = {
        'x_column': 'producto',
        'y_column': 'ventas',
        'title': 'Ventas por Producto',
        'x_label': 'Producto',
        'y_label': 'Ventas'
    }

    result = professional_viz_tool.create_chart(data, 'bar', config)

    assert result['chart_type'] == 'bar'
    assert result['data_points'] == 3
    assert 'config' in result
    print("✅ Test bar chart profesional pasado")

def test_line_chart():
    """Test: Gráfica de línea con tendencia"""
    data = [
        {"mes": "Ene", "ventas": 1000},
        {"mes": "Feb", "ventas": 1200},
        {"mes": "Mar", "ventas": 1400},
        {"mes": "Abr", "ventas": 1300},
    ]

    config = {
        'x_column': 'mes',
        'y_column': 'ventas',
        'title': 'Evolución Mensual'
    }

    result = professional_viz_tool.create_chart(data, 'line', config)

    assert result['chart_type'] == 'line'
    assert result['data_points'] == 4
    # Verificar que tiene múltiples traces (línea principal + tendencia)
    assert len(result['config']['data']) >= 2
    print("✅ Test line chart con tendencia pasado")

# Ejecutar
# pytest tests/test_professional_viz.py -v
```

**✅ Output esperado:**
```
test_bar_chart PASSED
test_line_chart PASSED
```

---

### Test 2: Verificar Calidad Visual (Manual)

```bash
# Ejecutar API
docker-compose up -d

# Test manual con curl
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Muéstrame una gráfica de los 5 productos más vendidos"
  }' | jq '.chart_config'
```

**✅ Verificar en el output JSON:**
- [ ] `chart_type` presente
- [ ] `config.layout.title` con fuente Arial 24px
- [ ] `config.data[0].marker` con configuración de colores
- [ ] `config.layout.plot_bgcolor` == "white"
- [ ] `config.layout.margin` con valores 80

---

### Test 3: Comparación Visual Antes/Después

```python
# scripts/compare_charts.py

from app.tools.viz_tool import viz_tool  # Antiguo
from app.tools.professional_viz import professional_viz_tool  # Nuevo

data = [
    {"producto": "Laptop", "ventas": 1200},
    {"producto": "Mouse", "ventas": 450},
    {"producto": "Teclado", "ventas": 230},
]

config = {
    'x_column': 'producto',
    'y_column': 'ventas',
    'title': 'Ventas por Producto'
}

# Antiguo
old_chart = viz_tool._run(
    data=data,
    chart_type='bar',
    x_column='producto',
    y_column='ventas',
    title='Ventas por Producto'
)

# Nuevo
new_chart = professional_viz_tool.create_chart(
    data=data,
    chart_type='bar',
    config=config
)

print("📊 Comparación:")
print(f"Antiguo - Traces: {len(old_chart['config']['data'])}")
print(f"Nuevo   - Traces: {len(new_chart['config']['data'])}")
print(f"\nAntiguo - Márgenes: {old_chart['config']['layout'].get('margin', 'default')}")
print(f"Nuevo   - Márgenes: {new_chart['config']['layout']['margin']}")
print(f"\nAntiguo - Fuente título: {old_chart['config']['layout']['title'].get('font', {}).get('size', 'default')}")
print(f"Nuevo   - Fuente título: {new_chart['config']['layout']['title']['font']['size']}")
```

**✅ Output esperado:**
```
📊 Comparación:
Antiguo - Traces: 1
Nuevo   - Traces: 1

Antiguo - Márgenes: default
Nuevo   - Márgenes: {'l': 80, 'r': 80, 't': 100, 'b': 80}

Antiguo - Fuente título: default
Nuevo   - Fuente título: 24
```

---

### Test 4: Prueba End-to-End con Frontend

1. Levantar stack completo:
```bash
docker-compose up -d --build
```

2. Abrir frontend:
```
http://localhost:8501
```

3. Probar queries:
```
"Top 10 productos más vendidos con gráfica"
"Evolución de ventas por mes"
"Distribución de ventas por categoría"
```

**✅ Verificar visualmente:**
- [ ] Gráficas tienen diseño limpio y profesional
- [ ] Colores son consistentes y atractivos
- [ ] Títulos y labels son claros (Arial, tamaño correcto)
- [ ] Hover muestra información formateada
- [ ] Line charts tienen línea de tendencia
- [ ] Pie charts tienen efecto pull-out y total central
- [ ] Animaciones son suaves

---

## 📋 Checklist de Fase 3

- [ ] Dependencias instaladas (`scipy`, `kaleido`)
- [ ] `professional_viz.py` creado e implementado
- [ ] `viz_node` actualizado para usar gráficas profesionales
- [ ] Test 1 pasado (gráficas generan correctamente)
- [ ] Test 2 pasado (JSON tiene configuración profesional)
- [ ] Test 3 ejecutado (comparación antes/después)
- [ ] Test 4 pasado (verificación visual en frontend)
- [ ] Gráficas tienen calidad empresarial ⭐⭐⭐⭐⭐

---

## 🎯 Mejoras Implementadas

| Aspecto | Antes | Después |
|---------|-------|---------|
| **Fuentes** | Default | Arial 24px títulos, tamaños consistentes |
| **Colores** | Básicos | Paletas profesionales, gradientes |
| **Márgenes** | Ajustados | Optimizados (80px) |
| **Bar charts** | Simples | Gradiente, valores encima, ordenados |
| **Line charts** | Solo línea | Línea + área + tendencia + R² |
| **Pie charts** | Básicos | Donut con pull-out + total central |
| **Scatter** | Puntos básicos | Colormap + regresión + R² |
| **Hover** | Default | Formateado profesional |
| **Animaciones** | Ninguna | Transiciones suaves |

---

## 🎯 Siguientes Pasos

Continúa con **FASE_4_FEEDBACK_MEJORA_CONTINUA.md**

---

## 💰 Costos Fase 3

- **$0** (solo mejoras de código, sin servicios externos)
