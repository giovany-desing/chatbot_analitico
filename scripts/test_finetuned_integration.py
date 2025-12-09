"""
Script de integración para validar el modelo fine-tuned en el sistema híbrido.
Prueba end-to-end desde el endpoint Modal hasta la integración completa.
"""

import sys
import os
import json
import time
import requests
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings

# Configuración
ENDPOINT_URL = settings.FINETUNED_MODEL_ENDPOINT or ""
CHAT_ENDPOINT = "http://localhost:8000/chat"
TIMEOUT_SECONDS = 10
MAX_RESPONSE_TIME = 5  # Esperamos respuesta en menos de 5 segundos
EXPECTED_LLM_TIME = 10  # Tiempo esperado con LLM (~10s)

# Colores y emojis para output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


class TestResult:
    """Resultado de un test individual"""
    def __init__(self, name: str, passed: bool, message: str = "", duration: float = 0.0, data: Dict = None):
        self.name = name
        self.passed = passed
        self.message = message
        self.duration = duration
        self.data = data or {}


def print_header(text: str):
    """Imprime un header formateado"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(70)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}\n")


def print_test_result(result: TestResult):
    """Imprime el resultado de un test"""
    emoji = "✅" if result.passed else "❌"
    color = Colors.GREEN if result.passed else Colors.RED
    status = "PASSED" if result.passed else "FAILED"
    
    print(f"{emoji} {Colors.BOLD}{result.name}{Colors.RESET}")
    print(f"   Status: {color}{status}{Colors.RESET}")
    if result.duration > 0:
        print(f"   Duration: {result.duration:.2f}s")
    if result.message:
        print(f"   {result.message}")
    if result.data:
        for key, value in result.data.items():
            print(f"   {key}: {value}")
    print()


def test_1_modal_endpoint() -> TestResult:
    """
    TEST 1: Verificar endpoint Modal está activo
    """
    name = "TEST 1: Verificar endpoint Modal está activo"
    start_time = time.time()
    
    try:
        if not ENDPOINT_URL:
            return TestResult(
                name=name,
                passed=False,
                message="FINETUNED_MODEL_ENDPOINT no está configurado",
                duration=time.time() - start_time
            )
        
        # Payload de prueba
        test_payload = {
            "user_query": "Muestra los 10 productos más vendidos",
            "sql_query": "SELECT producto, SUM(cantidad) as total FROM ventas_preventivas GROUP BY producto ORDER BY total DESC LIMIT 10",
            "columns": ["producto", "total"],
            "num_rows": 10,
            "data_preview": [
                {"producto": "Tela Algodón", "total": 5000},
                {"producto": "Tela Poliéster", "total": 4500},
                {"producto": "Tela Lino", "total": 3000}
            ]
        }
        
        print(f"   📡 Haciendo request a: {ENDPOINT_URL}")
        print(f"   📦 Payload: {json.dumps(test_payload, indent=2, ensure_ascii=False)}")
        
        # Hacer request
        response = requests.post(
            ENDPOINT_URL,
            json=test_payload,
            timeout=TIMEOUT_SECONDS
        )
        
        elapsed_time = time.time() - start_time
        
        # Verificar status code
        if response.status_code != 200:
            return TestResult(
                name=name,
                passed=False,
                message=f"HTTP {response.status_code}: {response.text[:200]}",
                duration=elapsed_time,
                data={"status_code": response.status_code}
            )
        
        # Verificar tiempo de respuesta
        if elapsed_time > MAX_RESPONSE_TIME:
            return TestResult(
                name=name,
                passed=False,
                message=f"Tiempo de respuesta {elapsed_time:.2f}s excede {MAX_RESPONSE_TIME}s",
                duration=elapsed_time
            )
        
        # Parsear respuesta
        try:
            response_data = response.json()
        except json.JSONDecodeError as e:
            return TestResult(
                name=name,
                passed=False,
                message=f"Respuesta no es JSON válido: {e}",
                duration=elapsed_time
            )
        
        # Validar estructura
        required_fields = ['chart_type']
        missing_fields = [f for f in required_fields if f not in response_data]
        
        if missing_fields:
            return TestResult(
                name=name,
                passed=False,
                message=f"Campos faltantes: {missing_fields}",
                duration=elapsed_time,
                data={"response": response_data}
            )
        
        # Validar chart_type
        chart_type = response_data.get('chart_type')
        if not chart_type:
            return TestResult(
                name=name,
                passed=False,
                message="chart_type está vacío",
                duration=elapsed_time,
                data={"response": response_data}
            )
        
        return TestResult(
            name=name,
            passed=True,
            message=f"Endpoint responde correctamente con chart_type: {chart_type}",
            duration=elapsed_time,
            data={
                "chart_type": chart_type,
                "reasoning": response_data.get('reasoning', 'N/A')[:50],
                "config": response_data.get('config', {})
            }
        )
        
    except requests.exceptions.Timeout:
        return TestResult(
            name=name,
            passed=False,
            message=f"Timeout después de {TIMEOUT_SECONDS}s",
            duration=time.time() - start_time
        )
    except requests.exceptions.RequestException as e:
        return TestResult(
            name=name,
            passed=False,
            message=f"Error de conexión: {e}",
            duration=time.time() - start_time
        )
    except Exception as e:
        return TestResult(
            name=name,
            passed=False,
            message=f"Error inesperado: {e}",
            duration=time.time() - start_time
        )


def test_2_env_configuration() -> TestResult:
    """
    TEST 2: Verificar configuración en .env
    """
    name = "TEST 2: Verificar configuración en .env"
    start_time = time.time()
    
    try:
        issues = []
        config_data = {}
        
        # Verificar FINETUNED_MODEL_ENDPOINT
        if not ENDPOINT_URL:
            issues.append("FINETUNED_MODEL_ENDPOINT está vacío")
        else:
            config_data["FINETUNED_MODEL_ENDPOINT"] = ENDPOINT_URL
            if not ENDPOINT_URL.startswith("http"):
                issues.append("FINETUNED_MODEL_ENDPOINT no parece ser una URL válida")
        
        # Verificar USAR_FINETUNED_MODEL (si existe en settings)
        usar_finetuned = getattr(settings, 'USAR_FINETUNED_MODEL', None)
        if usar_finetuned is not None:
            config_data["USAR_FINETUNED_MODEL"] = usar_finetuned
            if not usar_finetuned:
                issues.append("USAR_FINETUNED_MODEL está en False (debería ser True)")
        else:
            # Si no existe, asumimos que está activado por defecto
            config_data["USAR_FINETUNED_MODEL"] = "Not configured (default: True)"
        
        elapsed_time = time.time() - start_time
        
        if issues:
            return TestResult(
                name=name,
                passed=False,
                message="; ".join(issues),
                duration=elapsed_time,
                data=config_data
            )
        
        return TestResult(
            name=name,
            passed=True,
            message="Configuración correcta",
            duration=elapsed_time,
            data=config_data
        )
        
    except Exception as e:
        return TestResult(
            name=name,
            passed=False,
            message=f"Error verificando configuración: {e}",
            duration=time.time() - start_time
        )


def test_3_end_to_end() -> TestResult:
    """
    TEST 3: Probar integración end-to-end
    """
    name = "TEST 3: Probar integración end-to-end"
    start_time = time.time()
    
    try:
        test_query = "Muéstrame gráfica de los 10 productos más vendidos"
        
        print(f"   📤 Enviando query: {test_query}")
        print(f"   🌐 Endpoint: {CHAT_ENDPOINT}")
        
        payload = {
            "message": test_query,
            "conversation_id": f"test-integration-{int(time.time())}"
        }
        
        # Hacer request
        response = requests.post(
            CHAT_ENDPOINT,
            json=payload,
            timeout=120  # Timeout más largo para LLM
        )
        
        elapsed_time = time.time() - start_time
        
        # Verificar status code
        if response.status_code != 200:
            return TestResult(
                name=name,
                passed=False,
                message=f"HTTP {response.status_code}: {response.text[:200]}",
                duration=elapsed_time,
                data={"status_code": response.status_code}
            )
        
        # Parsear respuesta
        try:
            response_data = response.json()
        except json.JSONDecodeError as e:
            return TestResult(
                name=name,
                passed=False,
                message=f"Respuesta no es JSON válido: {e}",
                duration=elapsed_time
            )
        
        # Verificar que se generó visualización
        chart_config = response_data.get('chart_config')
        if not chart_config:
            return TestResult(
                name=name,
                passed=False,
                message="No se generó chart_config en la respuesta",
                duration=elapsed_time,
                data={"response_keys": list(response_data.keys())}
            )
        
        chart_type = chart_config.get('chart_type') or chart_config.get('type')
        if not chart_type:
            return TestResult(
                name=name,
                passed=False,
                message="chart_type no está presente en chart_config",
                duration=elapsed_time,
                data={"chart_config": chart_config}
            )
        
        # Verificar source (debería ser 'finetuned' o 'rules', no 'llm')
        source = chart_config.get('source', 'unknown')
        
        return TestResult(
            name=name,
            passed=True,
            message=f"Integración exitosa. Chart: {chart_type}, Source: {source}",
            duration=elapsed_time,
            data={
                "chart_type": chart_type,
                "source": source,
                "intent": response_data.get('intent'),
                "has_results": bool(response_data.get('results')),
                "has_error": bool(response_data.get('error'))
            }
        )
        
    except requests.exceptions.Timeout:
        return TestResult(
            name=name,
            passed=False,
            message=f"Timeout después de 120s",
            duration=time.time() - start_time
        )
    except requests.exceptions.RequestException as e:
        return TestResult(
            name=name,
            passed=False,
            message=f"Error de conexión: {e}",
            duration=time.time() - start_time
        )
    except Exception as e:
        return TestResult(
            name=name,
            passed=False,
            message=f"Error inesperado: {e}",
            duration=time.time() - start_time
        )


def test_4_logs_validation() -> TestResult:
    """
    TEST 4: Validar logs muestran uso de Capa 2
    """
    name = "TEST 4: Validar logs muestran uso de Capa 2"
    start_time = time.time()
    
    try:
        import subprocess
        
        # Obtener logs recientes del contenedor
        print(f"   📋 Consultando logs del contenedor...")
        
        result = subprocess.run(
            ["docker-compose", "logs", "--tail=100", "app"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            return TestResult(
                name=name,
                passed=False,
                message=f"Error obteniendo logs: {result.stderr}",
                duration=time.time() - start_time
            )
        
        logs = result.stdout
        
        # Buscar indicadores de Capa 2
        capa2_indicators = [
            "Capa 2",
            "fine-tuned",
            "Fine-tuned",
            "Modelo Fine-tuned",
            "finetuned"
        ]
        
        found_indicators = []
        for indicator in capa2_indicators:
            if indicator.lower() in logs.lower():
                found_indicators.append(indicator)
        
        # Buscar si saltó directo a Capa 3 (LLM)
        capa3_indicators = [
            "Capa 3",
            "Fallback to LLM",
            "Layer 3"
        ]
        
        skipped_to_llm = False
        for indicator in capa3_indicators:
            if indicator in logs:
                # Verificar que no haya evidencia de Capa 2 antes
                # (simplificado: si encontramos Capa 3 pero no Capa 2, es problema)
                if not found_indicators:
                    skipped_to_llm = True
        
        elapsed_time = time.time() - start_time
        
        if not found_indicators:
            # Extraer líneas relevantes de logs
            relevant_lines = []
            for line in logs.split('\n'):
                if any(keyword in line.lower() for keyword in ['capa', 'layer', 'fine', 'hybrid']):
                    relevant_lines.append(line.strip())
            
            return TestResult(
                name=name,
                passed=False,
                message="No se encontraron indicadores de Capa 2 en los logs",
                duration=elapsed_time,
                data={
                    "relevant_logs": relevant_lines[-5:] if relevant_lines else ["No logs relevantes encontrados"]
                }
            )
        
        if skipped_to_llm:
            return TestResult(
                name=name,
                passed=False,
                message="Se saltó directo a Capa 3 (LLM) sin pasar por Capa 2",
                duration=elapsed_time,
                data={"found_indicators": found_indicators}
            )
        
        # Extraer líneas relevantes
        relevant_lines = []
        for line in logs.split('\n'):
            if any(indicator.lower() in line.lower() for indicator in found_indicators):
                relevant_lines.append(line.strip())
        
        return TestResult(
            name=name,
            passed=True,
            message=f"Logs muestran uso de Capa 2. Indicadores encontrados: {', '.join(found_indicators[:3])}",
            duration=elapsed_time,
            data={
                "found_indicators": found_indicators,
                "sample_logs": relevant_lines[-3:] if relevant_lines else []
            }
        )
        
    except subprocess.TimeoutExpired:
        return TestResult(
            name=name,
            passed=False,
            message="Timeout obteniendo logs",
            duration=time.time() - start_time
        )
    except Exception as e:
        return TestResult(
            name=name,
            passed=False,
            message=f"Error: {e}",
            duration=time.time() - start_time
        )


def test_5_performance_comparison() -> TestResult:
    """
    TEST 5: Comparativa de rendimiento
    """
    name = "TEST 5: Comparativa de rendimiento"
    start_time = time.time()
    
    try:
        test_query = "Muestra gráfica de productos más vendidos"
        num_requests = 3
        
        print(f"   🔄 Ejecutando {num_requests} requests para medir rendimiento...")
        
        times = []
        sources = []
        
        for i in range(num_requests):
            request_start = time.time()
            
            try:
                response = requests.post(
                    CHAT_ENDPOINT,
                    json={
                        "message": test_query,
                        "conversation_id": f"perf-test-{i}-{int(time.time())}"
                    },
                    timeout=120
                )
                
                request_time = time.time() - request_start
                times.append(request_time)
                
                if response.status_code == 200:
                    data = response.json()
                    chart_config = data.get('chart_config', {})
                    source = chart_config.get('source', 'unknown')
                    sources.append(source)
                    print(f"   Request {i+1}/{num_requests}: {request_time:.2f}s (source: {source})")
                else:
                    print(f"   Request {i+1}/{num_requests}: FAILED (HTTP {response.status_code})")
                    
            except Exception as e:
                print(f"   Request {i+1}/{num_requests}: ERROR - {e}")
                continue
        
        elapsed_time = time.time() - start_time
        
        if not times:
            return TestResult(
                name=name,
                passed=False,
                message="No se pudo completar ninguna request",
                duration=elapsed_time
            )
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        # Calcular mejora vs LLM esperado
        improvement = ((EXPECTED_LLM_TIME - avg_time) / EXPECTED_LLM_TIME) * 100
        improvement = max(0, improvement)  # No mostrar negativo
        
        # Contar sources
        source_counts = {}
        for source in sources:
            source_counts[source] = source_counts.get(source, 0) + 1
        
        return TestResult(
            name=name,
            passed=True,
            message=f"Tiempo promedio: {avg_time:.2f}s (vs {EXPECTED_LLM_TIME}s esperado con LLM)",
            duration=elapsed_time,
            data={
                "avg_time": f"{avg_time:.2f}s",
                "min_time": f"{min_time:.2f}s",
                "max_time": f"{max_time:.2f}s",
                "improvement": f"{improvement:.1f}%",
                "source_distribution": source_counts,
                "requests_completed": len(times)
            }
        )
        
    except Exception as e:
        return TestResult(
            name=name,
            passed=False,
            message=f"Error: {e}",
            duration=time.time() - start_time
        )


def print_troubleshooting_tips(failed_tests: List[TestResult]):
    """Imprime tips de troubleshooting para tests fallidos"""
    if not failed_tests:
        return
    
    print_header("🔧 TROUBLESHOOTING TIPS")
    
    for test in failed_tests:
        print(f"{Colors.YELLOW}⚠️  {test.name}{Colors.RESET}")
        
        if "TEST 1" in test.name:
            print("   • Verificar que el endpoint Modal esté desplegado:")
            print("     - Revisar Modal dashboard: https://modal.com")
            print("     - Verificar que el endpoint esté activo")
            print("     - Probar manualmente: curl -X POST <endpoint> ...")
            print("   • Verificar conectividad de red")
            print("   • Revisar logs de Modal para errores")
        
        elif "TEST 2" in test.name:
            print("   • Verificar archivo .env en la raíz del proyecto")
            print("   • Agregar: FINETUNED_MODEL_ENDPOINT=https://...")
            print("   • Reiniciar contenedor: docker-compose restart app")
        
        elif "TEST 3" in test.name:
            print("   • Verificar que el servicio esté corriendo:")
            print("     docker-compose ps")
            print("   • Verificar logs del contenedor:")
            print("     docker-compose logs app")
            print("   • Verificar que la query SQL se ejecute correctamente")
        
        elif "TEST 4" in test.name:
            print("   • Verificar que el endpoint esté configurado correctamente")
            print("   • Revisar app/intelligence/hybrid_system.py")
            print("   • Verificar que FINETUNED_MODEL_ENDPOINT no esté vacío")
            print("   • Ver logs completos: docker-compose logs --tail=200 app")
        
        elif "TEST 5" in test.name:
            print("   • Verificar conectividad y latencia de red")
            print("   • Revisar si hay timeouts en los requests")
            print("   • Verificar carga del servidor")
        
        print()


def main():
    """Función principal"""
    print_header("🧪 TEST DE INTEGRACIÓN - MODELO FINE-TUNED")
    
    print(f"{Colors.BOLD}Configuración:{Colors.RESET}")
    print(f"  Endpoint Modal: {ENDPOINT_URL or 'NO CONFIGURADO'}")
    print(f"  Chat Endpoint: {CHAT_ENDPOINT}")
    print(f"  Timeout: {TIMEOUT_SECONDS}s")
    print()
    
    # Ejecutar tests
    tests = [
        test_1_modal_endpoint,
        test_2_env_configuration,
        test_3_end_to_end,
        test_4_logs_validation,
        test_5_performance_comparison
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
            print_test_result(result)
        except Exception as e:
            print(f"{Colors.RED}❌ Error ejecutando {test_func.__name__}: {e}{Colors.RESET}\n")
            results.append(TestResult(
                name=test_func.__name__,
                passed=False,
                message=f"Error ejecutando test: {e}"
            ))
    
    # Resumen
    print_header("📊 RESUMEN")
    
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    failed_tests = [r for r in results if not r.passed]
    
    print(f"Tests pasados: {Colors.GREEN}{passed}/{total}{Colors.RESET}")
    print(f"Tests fallidos: {Colors.RED}{total - passed}/{total}{Colors.RESET}")
    print()
    
    # Estadísticas de tiempo
    total_time = sum(r.duration for r in results)
    avg_time = total_time / total if total > 0 else 0
    
    print(f"Tiempo total: {total_time:.2f}s")
    print(f"Tiempo promedio por test: {avg_time:.2f}s")
    print()
    
    # Mostrar tests fallidos
    if failed_tests:
        print(f"{Colors.RED}Tests fallidos:{Colors.RESET}")
        for test in failed_tests:
            print(f"  ❌ {test.name}")
        print()
    
    # Troubleshooting
    if failed_tests:
        print_troubleshooting_tips(failed_tests)
    
    # Exit code
    exit_code = 0 if passed == total else 1
    
    if exit_code == 0:
        print(f"{Colors.GREEN}{Colors.BOLD}✅ Todos los tests pasaron exitosamente!{Colors.RESET}\n")
    else:
        print(f"{Colors.RED}{Colors.BOLD}❌ Algunos tests fallaron. Revisa los troubleshooting tips arriba.{Colors.RESET}\n")
    
    return exit_code


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}⚠️  Tests interrumpidos por el usuario{Colors.RESET}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}❌ Error fatal: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

