# 🎯 FASE 1: Fine-Tuning del Modelo para Gráficas

## Objetivo
Entrenar un modelo Llama 3 8B especializado en decidir el tipo de gráfica correcto basándose en datos y consultas del usuario.

---

## 📦 Prerequisitos

- Cuenta de Google (para Colab)
- Cuenta de HuggingFace (gratis)
- Cuenta de Modal.com (gratis, $30 créditos)

---

## 🔧 Paso 1: Preparar Dataset de Entrenamiento

### 1.1 Crear archivo de datos

```bash
# En tu proyecto
touch training_data/viz_training.jsonl
```

### 1.2 Agregar ejemplos de entrenamiento

```json
# training_data/viz_training.jsonl

{"instruction": "Decide la gráfica apropiada", "input": "Query: Muestra ventas por mes\nDatos: 12 filas\nColumnas: ['mes', 'ventas']\nTipo_mes: fecha\nTipo_ventas: numérico", "output": "{\"chart_type\": \"line\", \"x_column\": \"mes\", \"y_column\": \"ventas\", \"title\": \"Ventas Mensuales\", \"reasoning\": \"Datos temporales requieren line chart para mostrar tendencia\"}"}

{"instruction": "Decide la gráfica apropiada", "input": "Query: Top 10 productos más vendidos\nDatos: 10 filas\nColumnas: ['producto', 'cantidad']\nTipo_producto: categórico\nTipo_cantidad: numérico", "output": "{\"chart_type\": \"bar\", \"x_column\": \"producto\", \"y_column\": \"cantidad\", \"title\": \"Top 10 Productos\", \"reasoning\": \"Comparación de categorías usa bar chart ordenado\"}"}

{"instruction": "Decide la gráfica apropiada", "input": "Query: Distribución de ventas por categoría\nDatos: 5 filas\nColumnas: ['categoria', 'total']\nTipo_categoria: categórico\nTipo_total: numérico", "output": "{\"chart_type\": \"pie\", \"x_column\": \"categoria\", \"y_column\": \"total\", \"title\": \"Distribución por Categoría\", \"reasoning\": \"Proporciones con pocas categorías usa pie chart\"}"}

{"instruction": "Decide la gráfica apropiada", "input": "Query: Relación entre precio y cantidad\nDatos: 50 filas\nColumnas: ['precio', 'cantidad']\nTipo_precio: numérico\nTipo_cantidad: numérico", "output": "{\"chart_type\": \"scatter\", \"x_column\": \"precio\", \"y_column\": \"cantidad\", \"title\": \"Precio vs Cantidad\", \"reasoning\": \"Dos variables numéricas requieren scatter plot para ver correlación\"}"}

{"instruction": "Decide la gráfica apropiada", "input": "Query: Gráfica de ventas por producto\nDatos: 15 filas\nColumnas: ['producto', 'ventas']\nTipo_producto: categórico\nTipo_ventas: numérico", "output": "{\"chart_type\": \"bar\", \"x_column\": \"producto\", \"y_column\": \"ventas\", \"title\": \"Ventas por Producto\", \"reasoning\": \"Comparación de múltiples productos usa bar chart\"}"}

{"instruction": "Decide la gráfica apropiada", "input": "Query: Evolución trimestral de ingresos\nDatos: 8 filas\nColumnas: ['trimestre', 'ingresos']\nTipo_trimestre: fecha\nTipo_ingresos: numérico", "output": "{\"chart_type\": \"line\", \"x_column\": \"trimestre\", \"y_column\": \"ingresos\", \"title\": \"Evolución Trimestral\", \"reasoning\": \"Serie temporal trimestral usa line chart\"}"}
```

**📝 Nota:** Necesitas crear al menos **500 ejemplos** de calidad para un buen fine-tuning. Los ejemplos deben cubrir:
- Line charts (datos temporales)
- Bar charts (comparaciones, top N)
- Pie charts (proporciones, ≤7 categorías)
- Scatter plots (correlaciones)
- Histograms (distribuciones)

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

### 2.5 Preparar dataset

```python
# Celda 4: Cargar datos
from datasets import load_dataset

# Subir tu archivo viz_training.jsonl a Colab
# Files > Upload (botón izquierda)

dataset = load_dataset('json', data_files='viz_training.jsonl', split='train')

print(f"✅ Dataset cargado: {len(dataset)} ejemplos")
print(f"Ejemplo: {dataset[0]}")
```

### 2.6 Formatear prompts

```python
# Celda 5: Formatear datos
alpaca_prompt = """A continuación hay una instrucción que describe una tarea, junto con una entrada que proporciona más contexto. Escribe una respuesta que complete apropiadamente la solicitud.

### Instrucción:
{}

### Entrada:
{}

### Respuesta:
{}"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched=True)

print("✅ Dataset formateado")
```

### 2.7 Entrenar modelo

```python
# Celda 6: Configurar entrenamiento
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
        num_train_epochs = 3,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

print("🚀 Iniciando entrenamiento...")
trainer_stats = trainer.train()

print("✅ Entrenamiento completado")
print(f"Tiempo: {trainer_stats.metrics['train_runtime']:.2f}s")
print(f"Loss final: {trainer_stats.metrics['train_loss']:.4f}")
```

**⏱️ Tiempo estimado:** 2-4 horas con GPU T4 gratuita

### 2.8 Guardar modelo

```python
# Celda 7: Guardar modelo
model.save_pretrained("llama3_8b_viz_expert")
tokenizer.save_pretrained("llama3_8b_viz_expert")

print("✅ Modelo guardado localmente")
```

### 2.9 Subir a HuggingFace

```python
# Celda 8: Subir a HuggingFace
from huggingface_hub import login

# Tu token de HuggingFace (crear en: https://huggingface.co/settings/tokens)
login(token="hf_tu_token_aqui")

model.push_to_hub("tu-usuario/llama3-8b-viz-expert", token="hf_tu_token_aqui")
tokenizer.push_to_hub("tu-usuario/llama3-8b-viz-expert", token="hf_tu_token_aqui")

print("✅ Modelo subido a HuggingFace")
print("URL: https://huggingface.co/tu-usuario/llama3-8b-viz-expert")
```

---

## 🚀 Paso 3: Deploy en Modal.com

### 3.1 Instalar Modal CLI

```bash
# En tu terminal local
pip install modal
modal setup
```

### 3.2 Crear archivo de deployment

```python
# modal_viz_model.py

import modal
import json

stub = modal.Stub("viz-expert-model")

# Imagen con dependencias
image = modal.Image.debian_slim().pip_install(
    "torch==2.1.0",
    "transformers==4.36.0",
    "peft==0.7.0",
    "accelerate==0.25.0",
    "bitsandbytes==0.41.0"
)

# Descargar modelo (se hace una sola vez)
@stub.function(
    image=image,
    gpu="T4",
    timeout=600,
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    model_name = "tu-usuario/llama3-8b-viz-expert"

    print(f"Cargando modelo: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_4bit=True
    )

    print("✅ Modelo cargado")
    return model, tokenizer

# Endpoint de predicción
@stub.function(
    image=image,
    gpu="T4",
    timeout=60,
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
@modal.web_endpoint(method="POST")
def predict(request: dict):
    import torch

    # Cargar modelo
    model, tokenizer = load_model()

    # Extraer prompt
    prompt = request.get("prompt", "")

    # Tokenizar
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Generar
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decodificar
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extraer solo la respuesta (después de "### Respuesta:")
    if "### Respuesta:" in result:
        result = result.split("### Respuesta:")[1].strip()

    return {"prediction": result}

# Para testing local
@stub.local_entrypoint()
def test():
    test_prompt = """A continuación hay una instrucción que describe una tarea, junto con una entrada que proporciona más contexto. Escribe una respuesta que complete apropiadamente la solicitud.

### Instrucción:
Decide la gráfica apropiada

### Entrada:
Query: Muestra ventas por mes
Datos: 12 filas
Columnas: ['mes', 'ventas']
Tipo_mes: fecha
Tipo_ventas: numérico

### Respuesta:
"""

    result = predict.remote({"prompt": test_prompt})
    print("Resultado:", result)
```

### 3.3 Configurar secretos en Modal

```bash
# Crear secret con tu token de HuggingFace
modal secret create huggingface-secret HUGGINGFACE_TOKEN=hf_tu_token_aqui
```

### 3.4 Deployar

```bash
# Deploy del modelo
modal deploy modal_viz_model.py

# Salida esperada:
# ✓ Created web function predict => https://tu-usuario--viz-expert-model-predict.modal.run
```

**🎉 Guarda la URL generada, la necesitarás en la Fase 2**

---

## 🧪 Plan de Pruebas

### Test 1: Verificar modelo en Colab

```python
# En Colab, después del entrenamiento
alpaca_prompt_test = """A continuación hay una instrucción que describe una tarea, junto con una entrada que proporciona más contexto. Escribe una respuesta que complete apropiadamente la solicitud.

### Instrucción:
Decide la gráfica apropiada

### Entrada:
Query: Top 5 productos más vendidos
Datos: 5 filas
Columnas: ['producto', 'cantidad']
Tipo_producto: categórico
Tipo_cantidad: numérico

### Respuesta:
"""

inputs = tokenizer([alpaca_prompt_test], return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.1,
    do_sample=False
)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

**✅ Output esperado:**
```json
{
  "chart_type": "bar",
  "x_column": "producto",
  "y_column": "cantidad",
  "title": "Top 5 Productos Más Vendidos",
  "reasoning": "Comparación de categorías con valores numéricos usa bar chart"
}
```

---

### Test 2: Verificar deployment en Modal

```bash
# En tu terminal local
curl -X POST https://tu-usuario--viz-expert-model-predict.modal.run \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A continuación hay una instrucción que describe una tarea, junto con una entrada que proporciona más contexto. Escribe una respuesta que complete apropiadamente la solicitud.\n\n### Instrucción:\nDecide la gráfica apropiada\n\n### Entrada:\nQuery: Gráfica de ventas mensuales\nDatos: 12 filas\nColumnas: [\"mes\", \"ventas\"]\nTipo_mes: fecha\nTipo_ventas: numérico\n\n### Respuesta:\n"
  }'
```

**✅ Output esperado:**
```json
{
  "prediction": "{\"chart_type\": \"line\", \"x_column\": \"mes\", \"y_column\": \"ventas\", \"title\": \"Ventas Mensuales\", \"reasoning\": \"Datos temporales mensuales requieren line chart\"}"
}
```

---

### Test 3: Benchmark de precisión

```python
# En Colab o local
test_cases = [
    {
        "input": "Query: Top 10\nDatos: 10 filas\nColumnas: ['producto', 'ventas']",
        "expected": "bar"
    },
    {
        "input": "Query: Evolución anual\nDatos: 5 filas\nColumnas: ['año', 'revenue']",
        "expected": "line"
    },
    {
        "input": "Query: Distribución por región\nDatos: 4 filas\nColumnas: ['region', 'total']",
        "expected": "pie"
    },
]

correct = 0
for test in test_cases:
    result = model.predict(test["input"])
    predicted_type = json.loads(result)["chart_type"]

    if predicted_type == test["expected"]:
        correct += 1
        print(f"✅ {test['input'][:30]}... -> {predicted_type}")
    else:
        print(f"❌ {test['input'][:30]}... -> {predicted_type} (esperado: {test['expected']})")

accuracy = (correct / len(test_cases)) * 100
print(f"\n📊 Precisión: {accuracy:.1f}%")
```

**✅ Precisión esperada:** ≥85%

---

## 📋 Checklist de Fase 1

- [ ] Dataset de 500+ ejemplos creado
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
1. Guarda la URL de tu endpoint de Modal
2. Anota el nombre de tu modelo en HuggingFace
3. Continúa con **FASE_2_SISTEMA_HIBRIDO.md**

---

## 💰 Costos

- Google Colab: **$0** (GPU gratis) o **$10/mes** (Colab Pro para más tiempo)
- HuggingFace: **$0** (hosting de modelo gratis)
- Modal.com: **$0** (primeros $30 gratis al mes)

**Total Fase 1: $0-10**
