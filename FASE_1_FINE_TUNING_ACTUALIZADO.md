# 🎯 FASE 1: Fine-Tuning del Modelo para Gráficas (ACTUALIZADO)

## Objetivo
Entrenar un modelo Llama 3 8B especializado en decidir el tipo de gráfica correcto basándose en datos y consultas del usuario.

---

## 📦 Prerequisitos

- Cuenta de Google (para Colab)
- Cuenta de HuggingFace (gratis)
- Cuenta de Modal.com (gratis, $30 créditos)

---

## 🔧 Paso 1: Dataset de Entrenamiento ✅ YA GENERADO

### 1.1 Dataset disponible

**¡El dataset ya está listo!** Se generó automáticamente con 500 ejemplos:

```bash
# Archivos en tu proyecto:
training_data_complete.jsonl  # 500 ejemplos (RECOMENDADO)
training_data.jsonl           # 100 ejemplos (alternativo)
```

### 1.2 Formato del dataset

Cada línea es un ejemplo en formato chat (compatible con Llama 3):

```json
{
  "messages": [
    {
      "role": "system",
      "content": "Eres un experto en visualización de datos para análisis de ventas textiles. Debes elegir el mejor tipo de gráfico basándote en la query del usuario y los datos SQL disponibles."
    },
    {
      "role": "user",
      "content": "Query: Muestra los 10 productos más vendidos\nSQL: SELECT producto, SUM(cantidad) as total FROM ventas_preventivas GROUP BY producto ORDER BY total DESC LIMIT 10\nColumnas: [producto, total]\nFilas: 10\nData preview: [{\"producto\": \"Tela Algodón\", \"total\": 5000}]"
    },
    {
      "role": "assistant",
      "content": "{\"chart_type\": \"bar\", \"reasoning\": \"Top 10 implica ranking. Bar chart es ideal para comparar cantidades.\", \"confidence\": 0.98}"
    }
  ]
}
```

### 1.3 Distribución del dataset (500 ejemplos)

- **200 ejemplos (40%)** → Bar charts (rankings, comparaciones)
- **150 ejemplos (30%)** → Line charts (series temporales)
- **100 ejemplos (20%)** → Pie charts (distribuciones)
- **30 ejemplos (6%)** → Scatter plots (correlaciones)
- **20 ejemplos (4%)** → Histograms (frecuencias)

### 1.4 Validar dataset (opcional)

```bash
# Ver estadísticas del dataset
python3 << 'EOF'
import json
from collections import Counter

chart_types = []
with open('training_data_complete.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        assistant_msg = data['messages'][2]['content']
        chart_type = json.loads(assistant_msg)['chart_type']
        chart_types.append(chart_type)

print(f"Total ejemplos: {len(chart_types)}")
print("\nDistribución:")
for chart_type, count in Counter(chart_types).most_common():
    pct = count * 100 / len(chart_types)
    print(f"  {chart_type:12} {count:3} ({pct:4.1f}%)")
EOF
```

---

## 🚀 Paso 2: Fine-Tuning en Google Colab

### 2.1 Crear Notebook en Colab

1. Ir a https://colab.research.google.com/
2. Crear nuevo notebook: `fine_tune_viz_model.ipynb`
3. Cambiar runtime: **Runtime > Change runtime type > GPU > T4**

### 2.2 Instalar dependencias

```python
# Celda 1: Instalar librerías
!pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install -q datasets transformers trl peft accelerate bitsandbytes
```

### 2.3 Cargar modelo base

```python
# Celda 2: Importar y configurar
from unsloth import FastLanguageModel
import torch

max_seq_length = 2048
dtype = None  # Auto-detect
load_in_4bit = True  # Usar quantización 4-bit

# Cargar Llama 3 8B optimizado
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

print("✅ Modelo cargado")
```

### 2.4 Configurar LoRA para fine-tuning

```python
# Celda 3: Configurar LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,  # LoRA rank
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

print("✅ LoRA configurado")
```

### 2.5 Subir y cargar dataset

```python
# Celda 4: Subir dataset
from google.colab import files
from datasets import load_dataset

# Subir archivo
print("📤 Sube el archivo training_data_complete.jsonl:")
uploaded = files.upload()

# Cargar dataset
dataset = load_dataset('json', data_files='training_data_complete.jsonl', split='train')

print(f"✅ Dataset cargado: {len(dataset)} ejemplos")
print(f"\nEjemplo 0:")
print(dataset[0])
```

### 2.6 Formatear prompts para Llama 3

```python
# Celda 5: Formatear datos en formato chat
def formatting_prompts_func(examples):
    texts = []
    for messages in examples["messages"]:
        # Convertir a formato de chat de Llama 3
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)

print("✅ Dataset formateado")
print(f"\nEjemplo formateado:")
print(dataset[0]["text"][:500] + "...")
```

### 2.7 Entrenar modelo

```python
# Celda 6: Configurar y ejecutar entrenamiento
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 3,  # 3 épocas
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",  # Desactivar wandb
    ),
)

print("🚀 Iniciando entrenamiento...")
print("⏱️  Tiempo estimado: 2-4 horas con T4")
print()

trainer_stats = trainer.train()

print("\n✅ Entrenamiento completado")
print(f"⏱️  Tiempo: {trainer_stats.metrics['train_runtime']:.2f}s ({trainer_stats.metrics['train_runtime']/60:.1f} min)")
print(f"📉 Loss final: {trainer_stats.metrics['train_loss']:.4f}")
```

**⏱️ Tiempo estimado:** 2-4 horas con GPU T4 gratuita

### 2.8 Guardar modelo localmente

```python
# Celda 7: Guardar modelo
model.save_pretrained("llama3_8b_viz_expert")
tokenizer.save_pretrained("llama3_8b_viz_expert")

print("✅ Modelo guardado en: llama3_8b_viz_expert/")
```

### 2.9 Subir a HuggingFace

```python
# Celda 8: Subir a HuggingFace
from huggingface_hub import login

# Tu token de HuggingFace (crear en: https://huggingface.co/settings/tokens)
HF_TOKEN = "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"  # REEMPLAZA CON TU TOKEN
HF_USERNAME = "egsamaca56"   # REEMPLAZA CON TU USUARIO

login(token=HF_TOKEN)

# Subir modelo
model.push_to_hub(
    f"{HF_USERNAME}/llama3-8b-viz-expert",
    token=HF_TOKEN
)

tokenizer.push_to_hub(
    f"{HF_USERNAME}/llama3-8b-viz-expert",
    token=HF_TOKEN
)

print("✅ Modelo subido a HuggingFace")
print(f"🔗 URL: https://huggingface.co/{HF_USERNAME}/llama3-8b-viz-expert")
```

---

## 🚀 Paso 3: Deploy en Modal.com

### 3.1 Instalar Modal CLI

```bash
# En tu terminal local (Mac)
pip install modal
modal setup
```

### 3.2 Crear archivo de deployment

**Archivo nuevo:** `modal_viz_model.py`

```python
import modal
import json

stub = modal.Stub("viz-expert-model")

# Imagen con dependencias
image = modal.Image.debian_slim().pip_install(
    "torch==2.1.2",
    "transformers==4.37.0",
    "peft==0.8.0",
    "accelerate==0.26.0",
    "bitsandbytes==0.42.0"
)

# Variable global para cachear modelo
model_cache = {}

@stub.function(
    image=image,
    gpu="T4",
    timeout=300,
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def load_model():
    """Carga el modelo (se ejecuta una vez y se cachea)"""
    if "model" in model_cache:
        return model_cache["model"], model_cache["tokenizer"]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    model_name = "tu-usuario/llama3-8b-viz-expert"  # REEMPLAZA

    print(f"⏳ Cargando modelo: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_4bit=True
    )

    model_cache["model"] = model
    model_cache["tokenizer"] = tokenizer

    print("✅ Modelo cargado y cacheado")
    return model, tokenizer

@stub.function(
    image=image,
    gpu="T4",
    timeout=60,
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
@modal.web_endpoint(method="POST")
def predict(request: dict):
    """Endpoint de predicción"""
    import torch

    # Cargar modelo
    model, tokenizer = load_model.remote()

    # Extraer datos del request
    user_query = request.get("user_query", "")
    sql_query = request.get("sql_query", "")
    columns = request.get("columns", [])
    num_rows = request.get("num_rows", 0)
    data_preview = request.get("data_preview", [])

    # Construir prompt
    user_content = f"""Query: {user_query}
SQL: {sql_query}
Columnas: {columns}
Filas: {num_rows}
Data preview: {json.dumps(data_preview[:3])}"""

    messages = [
        {
            "role": "system",
            "content": "Eres un experto en visualización de datos para análisis de ventas textiles. Debes elegir el mejor tipo de gráfico basándote en la query del usuario y los datos SQL disponibles."
        },
        {
            "role": "user",
            "content": user_content
        }
    ]

    # Aplicar template de chat
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenizar
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Generar
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decodificar
    result = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    # Parsear JSON
    try:
        prediction = json.loads(result)
    except:
        prediction = {"error": "Failed to parse", "raw": result}

    return prediction

# Para testing local
@stub.local_entrypoint()
def test():
    result = predict.remote({
        "user_query": "Muestra los 10 productos más vendidos",
        "sql_query": "SELECT producto, SUM(cantidad) as total FROM ventas_preventivas GROUP BY producto ORDER BY total DESC LIMIT 10",
        "columns": ["producto", "total"],
        "num_rows": 10,
        "data_preview": [{"producto": "Tela Algodón", "total": 5000}]
    })
    print("Resultado:", json.dumps(result, indent=2, ensure_ascii=False))
```

### 3.3 Configurar secretos en Modal

```bash
# Crear secret con tu token de HuggingFace
modal secret create huggingface-secret HUGGINGFACE_TOKEN=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
```

### 3.4 Deployar

```bash
# Test local primero
modal run modal_viz_model.py
https://modal.com/apps/egsamaca56/main/ap-SoIPHctSUfh1EQCUgI9yZf
# Deploy en producción
modal deploy modal_viz_model.py
https://egsamaca56--viz-expert-model-predict.modal.run
✓ Created objects.
├── 🔨 Created mount /Users/giovanysamaca/Desktop/chatbot_analitico/modal_viz_model.py
├── 🔨 Created function load_model.
└── 🔨 Created web function predict => https://egsamaca56--viz-expert-model-predict.modal.run
✓ App deployed in 2.773s! 🎉

```

**🎉 Guarda la URL generada, la necesitarás en la Fase 2**

---

## 🧪 Plan de Pruebas

### Test 1: Verificar modelo en Colab

```python
# En Colab, después del entrenamiento (Celda 9)
FastLanguageModel.for_inference(model)  # Activar modo inferencia

test_messages = [
    {
        "role": "system",
        "content": "Eres un experto en visualización de datos para análisis de ventas textiles."
    },
    {
        "role": "user",
        "content": "Query: Top 5 productos más vendidos\nSQL: SELECT producto, SUM(cantidad) as total FROM ventas_preventivas GROUP BY producto ORDER BY total DESC LIMIT 5\nColumnas: [producto, total]\nFilas: 5"
    }
]

prompt = tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.1)
result = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

print("Resultado:")
print(result)
```

**✅ Output esperado:**
```json
{
  "chart_type": "bar",
  "reasoning": "Top 5 implica ranking. Bar chart es ideal para comparar.",
  "confidence": 0.97
}
```

---

### Test 2: Verificar deployment en Modal

```bash
# En tu terminal local
curl -X POST https://tu-usuario--viz-expert-model-predict.modal.run \
  -H "Content-Type: application/json" \
  -d '{
    "user_query": "Evolución de ventas por mes",
    "sql_query": "SELECT mes, SUM(total) as revenue FROM ventas_preventivas GROUP BY mes",
    "columns": ["mes", "revenue"],
    "num_rows": 12,
    "data_preview": [{"mes": "2024-01", "revenue": 45000}]
  }'
```

**✅ Output esperado:**
```json
{
  "chart_type": "line",
  "reasoning": "Serie temporal mensual. Line chart muestra tendencias.",
  "confidence": 0.98
}
```

---

### Test 3: Benchmark de precisión

```python
# En Colab o local con el modelo deployado
import requests
import json

MODAL_URL = "https://tu-usuario--viz-expert-model-predict.modal.run"

test_cases = [
    {
        "input": {
            "user_query": "Top 10 productos",
            "sql_query": "SELECT producto, SUM(cantidad) FROM ventas_preventivas GROUP BY producto LIMIT 10",
            "columns": ["producto", "cantidad"],
            "num_rows": 10
        },
        "expected": "bar"
    },
    {
        "input": {
            "user_query": "Ventas por mes",
            "sql_query": "SELECT mes, SUM(total) FROM ventas_preventivas GROUP BY mes",
            "columns": ["mes", "total"],
            "num_rows": 12
        },
        "expected": "line"
    },
    {
        "input": {
            "user_query": "Distribución por categoría",
            "sql_query": "SELECT categoria, COUNT(*) FROM ventas_preventivas GROUP BY categoria",
            "columns": ["categoria", "count"],
            "num_rows": 4
        },
        "expected": "pie"
    },
]

correct = 0
for i, test in enumerate(test_cases, 1):
    response = requests.post(MODAL_URL, json=test["input"])
    result = response.json()
    predicted = result.get("chart_type", "unknown")

    if predicted == test["expected"]:
        correct += 1
        print(f"✅ Test {i}: {predicted}")
    else:
        print(f"❌ Test {i}: {predicted} (esperado: {test['expected']})")

accuracy = (correct / len(test_cases)) * 100
print(f"\n📊 Precisión: {accuracy:.1f}% ({correct}/{len(test_cases)})")
```

**✅ Precisión esperada:** ≥85%

---

## 📋 Checklist de Fase 1

- [x] Dataset de 500 ejemplos generado (training_data_complete.jsonl)
- [ ] Modelo fine-tuneado en Colab
- [ ] Modelo subido a HuggingFace
- [ ] Modal.com configurado
- [ ] Modelo deployado en Modal
- [ ] Test 1 pasado (predicción en Colab)
- [ ] Test 2 pasado (API de Modal funciona)
- [ ] Test 3 pasado (precisión ≥85%)

---

## 🎯 Siguientes Pasos

Una vez completada esta fase:
1. ✅ Guarda la URL de tu endpoint de Modal
2. ✅ Anota el nombre de tu modelo en HuggingFace
3. ➡️ Continúa con **FASE_2_SISTEMA_HIBRIDO.md**

---

## 💰 Costos

- Google Colab: **$0** (GPU T4 gratis) o **$10/mes** (Colab Pro para más tiempo)
- HuggingFace: **$0** (hosting de modelo gratis)
- Modal.com: **$0** (primeros $30 gratis al mes)

**Total Fase 1: $0-10**

---

## 📚 Documentación Adicional

- **TRAINING_DATA_README.md** - Detalles del dataset generado
- **DATA_GENERADA_RESUMEN.md** - Resumen de los 500 ejemplos
- **scripts/generate_training_data.py** - Script para generar más datos
