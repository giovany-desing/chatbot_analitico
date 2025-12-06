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

# Variable global para cachear modelo (a nivel de contenedor)
_model = None
_tokenizer = None

def _load_model_once():
    """Carga el modelo una sola vez y lo cachea globalmente"""
    global _model, _tokenizer
    
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

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
    
    # Configurar chat_template si no existe (necesario para Llama3)
    if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
        print("📥 Configurando chat_template para Llama3...")
        # Template simple para Llama3
        llama3_chat_template = "{% for message in messages %}{% if message['role'] == 'system' %}<|start_header_id|>system<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>\n{% elif message['role'] == 'user' %}<|start_header_id|>user<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>\n{% elif message['role'] == 'assistant' %}<|start_header_id|>assistant<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"
        tokenizer.chat_template = llama3_chat_template
        print("✅ Chat template configurado para Llama3")
    
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

    _model = model
    _tokenizer = tokenizer

    print("✅ Modelo cargado y cacheado")
    return model, tokenizer

@stub.function(
    image=image,
    gpu="T4",
    timeout=600,  # 10 minutos para permitir carga inicial del modelo
    keep_warm=1,  # Mantener 1 instancia caliente para evitar cold start
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def _predict_internal(user_query: str, sql_query: str, columns: list, num_rows: int, data_preview: list):
    """Función interna para hacer predicciones (sin endpoint)"""
    import torch
    
    # Cargar modelo (se carga una vez y se cachea en memoria)
    # La primera vez puede tardar varios minutos
    print("🔄 Iniciando carga del modelo...")
    model, tokenizer = _load_model_once()
    print("✅ Modelo listo para predicción")

    # Construir prompt
    user_content = f"""Query: {user_query}
SQL: {sql_query}
Columnas: {columns}
Filas: {num_rows}
Data preview: {json.dumps(data_preview[:3])}"""

    messages = [
        {
            "role": "system",
            "content": "Eres un experto en visualización de datos para análisis de ventas textiles. Debes elegir el mejor tipo de gráfico basándote en la query del usuario y los datos SQL disponibles. Responde SOLO con un objeto JSON válido, sin texto adicional antes o después. El formato debe ser: {\"chart_type\": \"tipo\", \"reasoning\": \"razón\", \"config\": {...}}"
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
            max_new_tokens=512,  # Aumentado para asegurar que complete el JSON
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decodificar
    result = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    # Limpiar caracteres extraños y espacios
    result = result.strip()
    # Remover caracteres de control y caracteres no imprimibles
    result = ''.join(char for char in result if char.isprintable() or char in ['\n', '\t'])
    
    # Intentar extraer el primer JSON válido
    prediction = None
    
    # Método 1: Intentar parsear directamente
    try:
        prediction = json.loads(result)
    except json.JSONDecodeError:
        # Método 2: Buscar el primer objeto JSON válido
        import re
        # Buscar patrones de objetos JSON
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, result)
        
        for match in matches:
            try:
                prediction = json.loads(match)
                break  # Usar el primer JSON válido
            except json.JSONDecodeError:
                continue
        
        # Método 3: Si no se encontró, intentar extraer desde el primer {
        if prediction is None:
            first_brace = result.find('{')
            if first_brace != -1:
                # Buscar el último } que cierra el primer objeto
                brace_count = 0
                end_pos = first_brace
                for i in range(first_brace, len(result)):
                    if result[i] == '{':
                        brace_count += 1
                    elif result[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_pos = i + 1
                            break
                
                try:
                    json_str = result[first_brace:end_pos]
                    prediction = json.loads(json_str)
                except json.JSONDecodeError:
                    pass
    
    # Si aún no se pudo parsear, devolver error con información útil
    if prediction is None:
        # Extraer un preview del resultado para debugging
        preview = result[:500] if len(result) > 500 else result
        prediction = {
            "error": "Failed to parse JSON",
            "raw_preview": preview,
            "raw_length": len(result)
        }
    
    return prediction

@stub.function(
    image=image,
    gpu="T4",
    timeout=900,  # 15 minutos para permitir carga inicial del modelo
    keep_warm=1,  # Mantener 1 instancia caliente para evitar cold start
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
@modal.fastapi_endpoint(method="POST")
def predict(request: dict):
    """Endpoint de predicción"""
    import torch
    import traceback
    
    try:
        # Extraer datos del request
        user_query = request.get("user_query", "")
        sql_query = request.get("sql_query", "")
        columns = request.get("columns", [])
        num_rows = request.get("num_rows", 0)
        data_preview = request.get("data_preview", [])
        
        # Cargar modelo (se carga una vez y se cachea en memoria)
        # La primera vez puede tardar varios minutos
        print("🔄 Iniciando carga del modelo...")
        try:
            model, tokenizer = _load_model_once()
            print("✅ Modelo listo para predicción")
        except Exception as e:
            error_msg = f"Error cargando el modelo: {str(e)}\n{traceback.format_exc()}"
            print(f"❌ {error_msg}")
            return {
                "error": "Model loading failed",
                "message": str(e),
                "traceback": traceback.format_exc() if len(traceback.format_exc()) < 500 else "Traceback too long"
            }

        # Construir prompt
        user_content = f"""Query: {user_query}
SQL: {sql_query}
Columnas: {columns}
Filas: {num_rows}
Data preview: {json.dumps(data_preview[:3])}"""

        messages = [
            {
                "role": "system",
                "content": "Eres un experto en visualización de datos para análisis de ventas textiles. Debes elegir el mejor tipo de gráfico basándote en la query del usuario y los datos SQL disponibles. Responde SOLO con un objeto JSON válido, sin texto adicional antes o después. El formato debe ser: {\"chart_type\": \"tipo\", \"reasoning\": \"razón\", \"config\": {...}}"
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
        print("🔄 Generando predicción...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,  # Aumentado para asegurar que complete el JSON
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Decodificar
        result = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Limpiar caracteres extraños y espacios
        result = result.strip()
        # Remover caracteres de control y caracteres no imprimibles
        result = ''.join(char for char in result if char.isprintable() or char in ['\n', '\t'])
        
        # Intentar extraer el primer JSON válido
        prediction = None
        
        # Método 1: Intentar parsear directamente
        try:
            prediction = json.loads(result)
        except json.JSONDecodeError:
            # Método 2: Buscar el primer objeto JSON válido
            import re
            # Buscar patrones de objetos JSON
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.findall(json_pattern, result)
            
            for match in matches:
                try:
                    prediction = json.loads(match)
                    break  # Usar el primer JSON válido
                except json.JSONDecodeError:
                    continue
            
            # Método 3: Si no se encontró, intentar extraer desde el primer {
            if prediction is None:
                first_brace = result.find('{')
                if first_brace != -1:
                    # Buscar el último } que cierra el primer objeto
                    brace_count = 0
                    end_pos = first_brace
                    for i in range(first_brace, len(result)):
                        if result[i] == '{':
                            brace_count += 1
                        elif result[i] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_pos = i + 1
                                break
                    
                    try:
                        json_str = result[first_brace:end_pos]
                        prediction = json.loads(json_str)
                    except json.JSONDecodeError:
                        pass
        
        # Si aún no se pudo parsear, devolver error con información útil
        if prediction is None:
            # Extraer un preview del resultado para debugging
            preview = result[:500] if len(result) > 500 else result
            prediction = {
                "error": "Failed to parse JSON",
                "raw_preview": preview,
                "raw_length": len(result)
            }
        
        print("✅ Predicción completada")
        return prediction
        
    except Exception as e:
        error_msg = f"Error en el endpoint: {str(e)}\n{traceback.format_exc()}"
        print(f"❌ {error_msg}")
        return {
            "error": "Internal server error",
            "message": str(e),
            "traceback": traceback.format_exc()[:1000]  # Limitar tamaño del traceback
        }

# Para testing (ejecuta en el contenedor remoto)
@stub.local_entrypoint()
def test():
    # Usar .remote() para ejecutar en el contenedor de Modal donde están las dependencias
    result = _predict_internal.remote(
        user_query="Muestra los 10 productos más vendidos",
        sql_query="SELECT producto, SUM(cantidad) as total FROM ventas_preventivas GROUP BY producto ORDER BY total DESC LIMIT 10",
        columns=["producto", "total"],
        num_rows=10,
        data_preview=[{"producto": "Tela Algodón", "total": 5000}]
    )
    print("Resultado:", json.dumps(result, indent=2, ensure_ascii=False))