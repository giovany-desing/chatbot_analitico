"""
Script de testing para notificaciones de alertas.
Permite probar Slack y Email con diferentes severidades.
"""

import sys
import os
from pathlib import Path

# Agregar el directorio raíz del proyecto al PYTHONPATH
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
from datetime import datetime
from app.metrics.alerts import Alert
from app.metrics.notifications import send_notification, send_slack_notification, send_email_notification


def create_test_alert(severity: str = "critical") -> Alert:
    """Crea una alerta de prueba"""
    return Alert(
        severity=severity,
        component="router",
        metric="success_rate",
        message=f"Test alerta {severity}: Success rate bajo en router",
        current_value=0.82,
        threshold=0.85,
        previous_value=0.90,
        recommendation="Revisar cambios recientes en código o datos de entrenamiento",
        created_at=datetime.now()
    )


def test_slack(severity: str = "critical"):
    """Test notificación Slack"""
    print(f"\n{'='*60}")
    print(f"🧪 Testing Slack Notification ({severity})")
    print(f"{'='*60}\n")
    
    alert = create_test_alert(severity)
    result = send_slack_notification(alert)
    
    if result:
        print("✅ Notificación Slack enviada exitosamente")
    else:
        print("❌ Falló envío de notificación Slack")
        print("   Verifica:")
        print("   - SLACK_WEBHOOK_URL está configurado en .env")
        print("   - La URL del webhook es válida")
    
    return result


def test_email(severity: str = "critical"):
    """Test notificación Email"""
    print(f"\n{'='*60}")
    print(f"🧪 Testing Email Notification ({severity})")
    print(f"{'='*60}\n")
    
    alert = create_test_alert(severity)
    result = send_email_notification(alert)
    
    if result:
        print("✅ Notificación Email enviada exitosamente")
    else:
        print("❌ Falló envío de notificación Email")
        print("   Verifica:")
        print("   - EMAIL_ENABLED=true en .env")
        print("   - SMTP_HOST, SMTP_USERNAME, SMTP_PASSWORD configurados")
        print("   - EMAIL_FROM y EMAIL_TO configurados")
    
    return result


def test_all_channels(severity: str = "critical"):
    """Test todos los canales"""
    print(f"\n{'='*60}")
    print(f"🧪 Testing All Channels ({severity})")
    print(f"{'='*60}\n")
    
    alert = create_test_alert(severity)
    results = send_notification(alert)
    
    print("\n📊 Resultados:")
    for channel, success in results.items():
        status = "✅" if success else "❌"
        print(f"   {status} {channel}: {'Éxito' if success else 'Falló'}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Test notificaciones de alertas"
    )
    parser.add_argument(
        "--channel",
        choices=["slack", "email", "all"],
        default="all",
        help="Canal a testear"
    )
    parser.add_argument(
        "--severity",
        choices=["critical", "warning", "info"],
        default="critical",
        help="Severidad de la alerta de prueba"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🧪 TEST DE NOTIFICACIONES")
    print("="*60)
    
    try:
        if args.channel == "slack":
            test_slack(args.severity)
        elif args.channel == "email":
            test_email(args.severity)
        else:  # all
            test_all_channels(args.severity)
        
        print("\n" + "="*60)
        print("✅ Test completado")
        print("="*60 + "\n")
        
    except KeyboardInterrupt:
        print("\n⚠️  Test cancelado por el usuario")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error en test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

