# 📊 Chatbot Analítico - Sistema Completo con Streamlit

Sistema de chatbot analítico inteligente que permite hacer preguntas en lenguaje natural sobre datos de ventas, con generación automática de SQL, cálculo de KPIs y visualizaciones interactivas.

## 🎯 Características

- 💬 **Chat en lenguaje natural** - Pregunta en español sin escribir SQL
- 📊 **Gráficos interactivos** - Visualizaciones automáticas con Plotly
- 📈 **KPIs automáticos** - Revenue, ticket promedio, y más
- 🧠 **RAG con pgvector** - Mejora queries SQL con búsqueda semántica
- ⚡ **Caché inteligente** - Redis para respuestas instantáneas
- 🎨 **Frontend Streamlit** - Interfaz web profesional lista para usar

## 🚀 Quick Start (3 pasos)

### 1. Configurar credenciales

```bash
cp .env.example .env
nano .env  # Agregar tus credenciales
```

### 2. Ejecutar el script de setup automático

```bash
./setup.sh
```

### 3. ¡Listo! Abre tu navegador

```
http://localhost:8501
```

**O manualmente:**

```bash
docker-compose up -d --build
```

## 📦 Lo que se levanta automáticamente

| Servicio | Puerto | Descripción |
|----------|--------|-------------|
| **Frontend** | 8501 | Interfaz Streamlit |
| **API** | 8000 | FastAPI + LangChain |
| **Redis** | 6379 | Caché de queries |
| **PostgreSQL** | 5432 | Vector store (RAG) |

## 🎨 Frontend Features

### Chat Interface
![Streamlit Chat](https://via.placeholder.com/800x400?text=Streamlit+Chat+Interface)

- ✅ Historial de conversación persistente
- ✅ Gráficos interactivos renderizados en tiempo real
- ✅ KPIs en formato visual (cards)
- ✅ Tablas de datos expandibles
- ✅ SQL generado visible para debugging
- ✅ Ejemplos precargados
- ✅ Health check de servicios

### Sidebar
- Estado de conexión con la API
- Ejemplos clicables
- Configuración de visualización
- Botón de limpiar caché

## 💬 Ejemplos de Preguntas

### SQL Simple
```
¿Cuántas ventas preventivas hay?
Muéstrame los productos
```

### Con Gráficos
```
Gráfica de los 10 productos más vendidos
Muéstrame las ventas por mes en un gráfico
```

### KPIs
```
Calcula el revenue total
¿Cuál es el ticket promedio?
```

### Análisis Híbrido
```
Analiza las ventas del último mes con KPIs y gráfica
Dame un reporte completo de productos
```

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────┐
│         Frontend (Streamlit)            │
│         Port: 8501                      │
│    - Chat Interface                     │
│    - Plotly Charts                      │
│    - KPI Cards                          │
└──────────────┬──────────────────────────┘
               │ HTTP
               ↓
┌─────────────────────────────────────────┐
│         API (FastAPI)                   │
│         Port: 8000                      │
│    - LangGraph Workflow                 │
│    - LangChain Tools                    │
│    - Groq LLM (Llama 3.3 70B)          │
└─┬─────────┬──────────┬─────────────────┘
  │         │          │
  ↓         ↓          ↓
┌──────┐ ┌─────┐ ┌────────────┐
│Redis │ │Postgres│ │MySQL RDS │
│Cache │ │pgvector│ │ (AWS)    │
└──────┘ └─────┘ └────────────┘
```

## 📚 Documentación

- [QUICK_START.md](QUICK_START.md) - Guía detallada paso a paso
- [FRONTEND_README.md](FRONTEND_README.md) - Documentación del frontend
- [FRONTEND_OPTIONS.md](FRONTEND_OPTIONS.md) - Alternativas de frontend
- [API Docs](http://localhost:8000/docs) - Swagger UI (cuando esté corriendo)

## 🔧 Desarrollo

### Estructura del proyecto

```
chatbot_analitico/
├── app/
│   ├── main.py              # FastAPI server
│   ├── agents/              # LangGraph workflow
│   ├── db/                  # Database connections
│   ├── llm/                 # LLM models
│   ├── rag/                 # RAG system
│   ├── services/            # Cache service
│   └── tools/               # SQL & Viz tools
├── front_app.py             # Streamlit frontend ⭐
├── front_gradio.py          # Gradio alternative
├── front_notebook.ipynb     # Jupyter notebook
├── docker-compose.yml       # Stack completo
├── Dockerfile               # API image
├── Dockerfile.frontend      # Frontend image
├── setup.sh                 # Setup script
└── requirements.txt         # Python dependencies
```

### Comandos útiles

```bash
# Ver logs
docker-compose logs -f

# Ver logs solo del frontend
docker-compose logs -f frontend

# Reiniciar frontend después de cambios
docker-compose restart frontend

# Reconstruir tras cambios en código
docker-compose up -d --build frontend

# Detener todo
docker-compose down

# Ver uso de recursos
docker stats
```

### Desarrollo local del frontend (sin Docker)

```bash
# Instalar dependencias
pip install streamlit plotly pandas requests

# Configurar API URL
export API_URL=http://localhost:8000

# Ejecutar
streamlit run front_app.py
```

## 🐛 Troubleshooting

### ❌ "API no disponible"

```bash
# Ver logs de la API
docker-compose logs -f app

# Verificar health
curl http://localhost:8000/health

# Reiniciar API
docker-compose restart app
```

### ❌ Error de conexión a MySQL

1. Verifica credenciales en `.env`
2. Verifica que RDS sea públicamente accesible
3. Verifica Security Groups en AWS
4. Prueba conexión manual:
   ```bash
   mysql -h your-endpoint.rds.amazonaws.com -u admin -p
   ```

### ❌ Puerto 8501 en uso

```bash
# Ver qué está usando el puerto
lsof -i :8501

# Matar proceso
kill -9 <PID>

# O cambiar puerto en docker-compose.yml
```

## 📊 Stack Tecnológico

### Backend
- **FastAPI** - API REST
- **LangChain** - Framework LLM
- **LangGraph** - Orquestación de agentes
- **Groq** - Llama 3.3 70B
- **SQLAlchemy** - ORM

### Frontend
- **Streamlit** - Web UI
- **Plotly** - Gráficos interactivos
- **Pandas** - Manipulación de datos

### Bases de Datos
- **MySQL (AWS RDS)** - Datos transaccionales
- **PostgreSQL + pgvector** - Vector store
- **Redis** - Caché

### DevOps
- **Docker** - Containerización
- **Docker Compose** - Orquestación

## 🔐 Seguridad

⚠️ **Importante para producción:**

- [ ] No expongas puertos innecesarios
- [ ] Usa HTTPS con certificados SSL
- [ ] Configura autenticación
- [ ] Usa secretos seguros (no en código)
- [ ] Configura rate limiting
- [ ] Actualiza dependencias regularmente

## 📈 Roadmap

- [ ] Autenticación de usuarios
- [ ] Historial persistente de conversaciones
- [ ] Exportar reportes en PDF
- [ ] Soporte para más tipos de gráficos
- [ ] Múltiples idiomas
- [ ] Integración con Slack/Teams
- [ ] Dashboard de analytics
- [ ] A/B testing de modelos LLM

## 🤝 Contribuir

1. Fork el proyecto
2. Crea tu feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push al branch (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto está bajo la licencia MIT.

## 🙏 Agradecimientos

- [LangChain](https://langchain.com/) - Framework LLM
- [Streamlit](https://streamlit.io/) - Framework web
- [Groq](https://groq.com/) - Inference rápida
- [Plotly](https://plotly.com/) - Visualizaciones

## 📧 Soporte

¿Problemas o preguntas?

1. Revisa [QUICK_START.md](QUICK_START.md)
2. Revisa los logs: `docker-compose logs -f`
3. Abre un issue en GitHub

---

**Hecho con ❤️ usando LangChain, Streamlit y FastAPI**

## 🎉 ¡Empieza Ahora!

```bash
# Un solo comando para levantar todo:
./setup.sh

# O manualmente:
docker-compose up -d --build

# Abre tu navegador:
# http://localhost:8501
```

**¡Disfruta del chatbot!** 🚀📊
