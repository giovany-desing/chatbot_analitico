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
