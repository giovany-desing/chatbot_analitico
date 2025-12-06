import modal
import json
import os


stub = modal.App("viz-expert-model")
app = stub  # Alias para modal deploy 

# Imagen con dependencias
image = modal.Image.debian_slim().pip_install(
    "fastapi[standard]",
    "numpy",
    "torch",  # Versión más reciente compatible con transformers>=4.40.0
    "transformers",  # Versión más reciente compatible con peft>=0.9.0
    "peft",  # Versión más reciente para soportar alora_invocation_tokens
    "accelerate",  # Versión más reciente compatible con peft>=0.9.0
    "bitsandbytes"
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
    import tempfile
    from peft import PeftModel
    from huggingface_hub import hf_hub_download

    model_name = "egsamaca56/llama3-8b-viz-expert" 

    print(f"⏳ Cargando modelo fine-tuned: {model_name}")

    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise ValueError("HUGGINGFACE_TOKEN environment variable not set. Make sure the Modal secret 'huggingface-secret' is configured correctly.")
    
    # Cargar tokenizer desde modelo base (el tokenizer no cambia en fine-tuning)
    # El tokenizer.json del modelo fine-tuned está corrupto, así que usamos el modelo base original
    print("📥 Cargando tokenizer desde modelo base (unsloth/llama-3-8b-bnb-4bit)...")
    base_model_name = "unsloth/llama-3-8b-bnb-4bit"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, 
            token=hf_token, 
            use_fast=False
        )
        print("✅ Tokenizer cargado desde modelo base")
    except Exception as e:
        print(f"❌ Error cargando tokenizer desde {base_model_name}: {e}")
        # Si falla, intentar desde el modelo fine-tuned como último recurso
        print("📥 Intentando cargar tokenizer desde modelo fine-tuned como último recurso...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=hf_token,
                use_fast=False,
                trust_remote_code=True
            )
            print("✅ Tokenizer cargado desde modelo fine-tuned")
        except Exception as e2:
            raise RuntimeError(
                f"No se pudo cargar el tokenizer. El archivo tokenizer.json del modelo "
                f"{model_name} parece estar corrupto. Por favor, verifica el repositorio en Hugging Face."
            ) from e2
    
    # Configurar pad_token si no existe (necesario para algunos modelos de Llama)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print("🔧 Pad token configurado")
    
    # Cargar modelo fine-tuned (SIEMPRE desde el repositorio fine-tuned)
    print(f"📥 Cargando modelo fine-tuned desde {model_name}...")
    
    try:
        # Intentar cargar directamente el modelo fine-tuned
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_4bit=True,
            token=hf_token,
            trust_remote_code=True
        )
        print("✅ Modelo fine-tuned cargado directamente")
    except (TypeError, ValueError) as e:
        if "alora_invocation_tokens" in str(e) or "unexpected keyword argument" in str(e):
            print("⚠️ Error de compatibilidad con configuración LoRA, cargando adaptadores del modelo fine-tuned...")
            
            try:
                # Descargar y limpiar la configuración LoRA del modelo fine-tuned
                config_path = hf_hub_download(
                    repo_id=model_name,
                    filename="adapter_config.json",
                    token=hf_token
                )
                
                # Leer y remover parámetros desconocidos
                with open(config_path, 'r') as f:
                    config = json.load(f)
                config.pop('alora_invocation_tokens', None)
                
                # Guardar configuración limpia
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                    json.dump(config, tmp)
                    temp_config_path = tmp.name
                
                # Cargar modelo base (necesario para cargar adaptadores LoRA)
                base_model_name = "unsloth/llama-3-8b-bnb-4bit"
                print(f"📥 Cargando modelo base desde {base_model_name}...")
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    load_in_4bit=True,
                    token=hf_token
                )
                
                # IMPORTANTE: Cargar adaptadores LoRA desde el modelo fine-tuned
                print(f"📥 Cargando adaptadores LoRA desde {model_name} (modelo fine-tuned)...")
                model = PeftModel.from_pretrained(
                    model,
                    model_name,  # SIEMPRE desde el modelo fine-tuned
                    token=hf_token
                )
                
                os.unlink(temp_config_path)
                print(f"✅ Modelo fine-tuned cargado: base + adaptadores LoRA desde {model_name}")
            except Exception as e2:
                raise RuntimeError(
                    f"Error cargando el modelo fine-tuned desde {model_name}. Error: {e2}"
                ) from e2
        else:
            raise RuntimeError(
                f"Error cargando el modelo fine-tuned desde {model_name}. Error: {e}"
            ) from e

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
@modal.fastapi_endpoint(method="POST")
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
    result = predict.local({
        "user_query": "Muestra los 10 productos más vendidos",
        "sql_query": "SELECT producto, SUM(cantidad) as total FROM ventas_preventivas GROUP BY producto ORDER BY total DESC LIMIT 10",
        "columns": ["producto", "total"],
        "num_rows": 10,
        "data_preview": [{"producto": "Tela Algodón", "total": 5000}]
    })
    print("Resultado:", json.dumps(result, indent=2, ensure_ascii=False))