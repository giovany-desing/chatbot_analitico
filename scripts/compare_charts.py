import sys
import os
from pathlib import Path

# Agregar el directorio raíz al PYTHONPATH
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

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