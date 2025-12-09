"""
Sistema de notificaciones para alertas críticas.
Soporta Slack y Email con rate limiting y manejo robusto de errores.
"""

import logging
import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import requests
from sqlalchemy import text

from app.config import settings
from app.db.connections import get_postgres
from app.metrics.alerts import Alert

logger = logging.getLogger(__name__)

# ============ Configuración ============

NOTIFICATIONS_ENABLED = getattr(settings, 'NOTIFICATIONS_ENABLED', False)
NOTIFICATION_CHANNELS = getattr(settings, 'NOTIFICATION_CHANNELS', 'slack,email').split(',')

# Rate limiting
MAX_CRITICAL_PER_HOUR = 5
COOLDOWN_HOURS = 1

# ============ Crear Tabla de Historial ============

def create_alert_history_table() -> None:
    """Crea la tabla alert_history si no existe"""
    try:
        postgres = get_postgres()
        session = postgres.get_session()
        
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS alert_history (
            id SERIAL PRIMARY KEY,
            component VARCHAR(50) NOT NULL,
            metric VARCHAR(50) NOT NULL,
            severity VARCHAR(20) NOT NULL,
            sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            channels JSONB,
            delivery_status JSONB,
            alert_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_alert_component_metric ON alert_history(component, metric);
        CREATE INDEX IF NOT EXISTS idx_alert_sent_at ON alert_history(sent_at DESC);
        CREATE INDEX IF NOT EXISTS idx_alert_severity ON alert_history(severity);
        """
        
        session.execute(text(create_table_sql))
        session.commit()
        session.close()
        
        logger.debug("✅ Tabla alert_history creada/verificada")
        
    except Exception as e:
        logger.error(f"❌ Error creando tabla alert_history: {e}", exc_info=True)


def create_pending_alerts_table() -> None:
    """Crea la tabla pending_alerts para alertas no enviadas"""
    try:
        postgres = get_postgres()
        session = postgres.get_session()
        
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS pending_alerts (
            id SERIAL PRIMARY KEY,
            alert_data JSONB NOT NULL,
            channels JSONB NOT NULL,
            attempts INT DEFAULT 0,
            last_attempt_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            next_retry_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_pending_next_retry ON pending_alerts(next_retry_at);
        """
        
        session.execute(text(create_table_sql))
        session.commit()
        session.close()
        
        logger.debug("✅ Tabla pending_alerts creada/verificada")
        
    except Exception as e:
        logger.error(f"❌ Error creando tabla pending_alerts: {e}", exc_info=True)


# Inicializar tablas al importar
try:
    create_alert_history_table()
    create_pending_alerts_table()
except Exception as e:
    logger.warning(f"⚠️ Error inicializando tablas de notificaciones: {e}")


# ============ Rate Limiting ============

def check_rate_limit(alert: Alert) -> bool:
    """
    Verifica si la alerta puede ser enviada según rate limiting.
    
    Args:
        alert: Alert a verificar
    
    Returns:
        True si puede enviarse, False si está en cooldown
    """
    try:
        postgres = get_postgres()
        session = postgres.get_session()
        
        # Para alertas críticas, verificar máximo por hora
        if alert.severity == "critical":
            one_hour_ago = datetime.now() - timedelta(hours=1)
            
            sql = """
            SELECT COUNT(*) as count
            FROM alert_history
            WHERE component = :component
                AND metric = :metric
                AND severity = 'critical'
                AND sent_at >= :one_hour_ago
            """
            
            result = session.execute(
                text(sql),
                {
                    'component': alert.component,
                    'metric': alert.metric,
                    'one_hour_ago': one_hour_ago
                }
            ).fetchone()
            
            count = result[0] if result else 0
            
            if count >= MAX_CRITICAL_PER_HOUR:
                logger.warning(f"⚠️ Rate limit alcanzado para {alert.component}/{alert.metric}: {count} alertas en última hora")
                session.close()
                return False
        
        # Verificar cooldown (1 hora entre alertas similares)
        cooldown_time = datetime.now() - timedelta(hours=COOLDOWN_HOURS)
        
        sql = """
        SELECT sent_at
        FROM alert_history
        WHERE component = :component
            AND metric = :metric
            AND severity = :severity
            AND sent_at >= :cooldown_time
        ORDER BY sent_at DESC
        LIMIT 1
        """
        
        result = session.execute(
            text(sql),
            {
                'component': alert.component,
                'metric': alert.metric,
                'severity': alert.severity,
                'cooldown_time': cooldown_time
            }
        ).fetchone()
        
        session.close()
        
        if result and result[0]:
            logger.info(f"⏸️  Alerta en cooldown: {alert.component}/{alert.metric} (última: {result[0]})")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error verificando rate limit: {e}", exc_info=True)
        # En caso de error, permitir envío (fail-open)
        return True


def record_alert_sent(alert: Alert, channels: Dict[str, bool]) -> None:
    """Registra que una alerta fue enviada"""
    try:
        postgres = get_postgres()
        session = postgres.get_session()
        
        insert_sql = """
        INSERT INTO alert_history (
            component,
            metric,
            severity,
            channels,
            delivery_status,
            alert_message
        ) VALUES (
            :component,
            :metric,
            :severity,
            :channels,
            :delivery_status,
            :alert_message
        )
        """
        
        session.execute(
            text(insert_sql),
            {
                'component': alert.component,
                'metric': alert.metric,
                'severity': alert.severity,
                'channels': json.dumps(list(channels.keys())),
                'delivery_status': json.dumps(channels),
                'alert_message': alert.message
            }
        )
        session.commit()
        session.close()
        
    except Exception as e:
        logger.error(f"❌ Error registrando alerta enviada: {e}", exc_info=True)


# ============ Notificaciones Slack ============

def format_slack_message(alert: Alert) -> Dict[str, Any]:
    """Formatea mensaje de alerta para Slack usando Blocks API"""
    
    # Determinar color y emoji según severidad
    severity_config = {
        "critical": {
            "color": "danger",
            "emoji": "🚨",
            "title": "ALERTA CRÍTICA"
        },
        "warning": {
            "color": "warning",
            "emoji": "⚠️",
            "title": "ALERTA DE ADVERTENCIA"
        },
        "info": {
            "color": "good",
            "emoji": "ℹ️",
            "title": "INFORMACIÓN"
        }
    }
    
    config = severity_config.get(alert.severity, severity_config["info"])
    
    # Formatear valor actual
    if isinstance(alert.current_value, float):
        if alert.metric in ['success_rate', 'correction_rate', 'llm_fallback_rate']:
            current_str = f"{alert.current_value:.1%}"
        else:
            current_str = f"{alert.current_value:.2f}"
    else:
        current_str = str(alert.current_value)
    
    # Formatear threshold si existe
    threshold_str = "N/A"
    if alert.threshold is not None:
        if isinstance(alert.threshold, float):
            if alert.metric in ['success_rate', 'correction_rate', 'llm_fallback_rate']:
                threshold_str = f"{alert.threshold:.1%}"
            else:
                threshold_str = f"{alert.threshold:.2f}"
        else:
            threshold_str = str(alert.threshold)
    
    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{config['emoji']} {config['title']}"
            }
        },
        {
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": f"*Componente:*\n{alert.component.upper()}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Métrica:*\n{alert.metric}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Valor Actual:*\n{current_str}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Umbral:*\n{threshold_str}"
                }
            ]
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Mensaje:*\n{alert.message}"
            }
        }
    ]
    
    # Agregar recomendación si existe
    if alert.recommendation:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Recomendación:*\n{alert.recommendation}"
            }
        })
    
    # Agregar timestamp
    blocks.append({
        "type": "context",
        "elements": [
            {
                "type": "mrkdwn",
                "text": f"Timestamp: {alert.created_at.strftime('%Y-%m-%d %H:%M:%S')}"
            }
        ]
    })
    
    return {
        "blocks": blocks,
        "username": getattr(settings, 'SLACK_USERNAME', 'Chatbot Monitor'),
        "icon_emoji": getattr(settings, 'SLACK_ICON_EMOJI', ':robot_face:'),
        "channel": getattr(settings, 'SLACK_CHANNEL', '#chatbot-alerts')
    }


def send_slack_notification(alert: Alert) -> bool:
    """
    Envía notificación a Slack.
    
    Args:
        alert: Alert a enviar
    
    Returns:
        True si éxito, False si falla
    """
    webhook_url = getattr(settings, 'SLACK_WEBHOOK_URL', None)
    
    if not webhook_url:
        logger.warning("⚠️ SLACK_WEBHOOK_URL no configurado")
        return False
    
    try:
        message = format_slack_message(alert)
        
        # Retry 2 veces
        for attempt in range(3):
            try:
                response = requests.post(
                    webhook_url,
                    json=message,
                    timeout=10,
                    headers={'Content-Type': 'application/json'}
                )
                
                response.raise_for_status()
                
                if attempt > 0:
                    logger.info(f"✅ Notificación Slack enviada en intento {attempt + 1}")
                
                return True
                
            except requests.exceptions.RequestException as e:
                if attempt < 2:
                    logger.warning(f"⚠️ Intento {attempt + 1} falló para Slack: {e}. Reintentando...")
                    continue
                else:
                    raise
        
        return False
        
    except Exception as e:
        logger.error(f"❌ Error enviando notificación Slack: {e}", exc_info=True)
        return False


# ============ Notificaciones Email ============

def format_email_html(alert: Alert) -> str:
    """Formatea mensaje de alerta como HTML para email"""
    
    severity_config = {
        "critical": {
            "color": "#dc3545",
            "title": "🚨 ALERTA CRÍTICA",
            "bg_color": "#f8d7da"
        },
        "warning": {
            "color": "#ffc107",
            "title": "⚠️ ALERTA DE ADVERTENCIA",
            "bg_color": "#fff3cd"
        },
        "info": {
            "color": "#17a2b8",
            "title": "ℹ️ INFORMACIÓN",
            "bg_color": "#d1ecf1"
        }
    }
    
    config = severity_config.get(alert.severity, severity_config["info"])
    
    # Formatear valores
    if isinstance(alert.current_value, float):
        if alert.metric in ['success_rate', 'correction_rate', 'llm_fallback_rate']:
            current_str = f"{alert.current_value:.1%}"
        else:
            current_str = f"{alert.current_value:.2f}"
    else:
        current_str = str(alert.current_value)
    
    threshold_str = "N/A"
    if alert.threshold is not None:
        if isinstance(alert.threshold, float):
            if alert.metric in ['success_rate', 'correction_rate', 'llm_fallback_rate']:
                threshold_str = f"{alert.threshold:.1%}"
            else:
                threshold_str = f"{alert.threshold:.2f}"
        else:
            threshold_str = str(alert.threshold)
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 600px;
                margin: 0 auto;
                padding: 20px;
            }}
            .header {{
                background-color: {config['bg_color']};
                padding: 20px;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
            .header h1 {{
                color: {config['color']};
                margin: 0;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #f2f2f2;
                font-weight: bold;
            }}
            .recommendation {{
                background-color: #e7f3ff;
                padding: 15px;
                border-left: 4px solid #2196F3;
                margin: 20px 0;
            }}
            .footer {{
                margin-top: 30px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
                font-size: 12px;
                color: #666;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>{config['title']}</h1>
        </div>
        
        <table>
            <tr>
                <th>Componente</th>
                <td>{alert.component.upper()}</td>
            </tr>
            <tr>
                <th>Métrica</th>
                <td>{alert.metric}</td>
            </tr>
            <tr>
                <th>Valor Actual</th>
                <td><strong>{current_str}</strong></td>
            </tr>
            <tr>
                <th>Umbral</th>
                <td>{threshold_str}</td>
            </tr>
            <tr>
                <th>Severidad</th>
                <td><strong>{alert.severity.upper()}</strong></td>
            </tr>
            <tr>
                <th>Timestamp</th>
                <td>{alert.created_at.strftime('%Y-%m-%d %H:%M:%S')}</td>
            </tr>
        </table>
        
        <h3>Mensaje</h3>
        <p>{alert.message}</p>
        
        {f'<div class="recommendation"><strong>Recomendación:</strong><br>{alert.recommendation}</div>' if alert.recommendation else ''}
        
        <div class="footer">
            <p>Este es un mensaje automático del sistema de monitoreo de Chatbot Analítico.</p>
            <p>Para más información, consulta el dashboard de métricas.</p>
        </div>
    </body>
    </html>
    """
    
    return html


def send_email_notification(alert: Alert) -> bool:
    """
    Envía notificación por email.
    
    Args:
        alert: Alert a enviar
    
    Returns:
        True si éxito, False si falla
    """
    if not getattr(settings, 'EMAIL_ENABLED', False):
        logger.warning("⚠️ Email deshabilitado")
        return False
    
    smtp_host = getattr(settings, 'SMTP_HOST', None)
    smtp_port = getattr(settings, 'SMTP_PORT', 587)
    smtp_username = getattr(settings, 'SMTP_USERNAME', None)
    smtp_password = getattr(settings, 'SMTP_PASSWORD', None)
    email_from = getattr(settings, 'EMAIL_FROM', None)
    email_to = getattr(settings, 'EMAIL_TO', '').split(',')
    subject_prefix = getattr(settings, 'EMAIL_SUBJECT_PREFIX', '[CHATBOT ALERT]')
    
    if not all([smtp_host, smtp_username, smtp_password, email_from, email_to]):
        logger.warning("⚠️ Configuración de email incompleta")
        return False
    
    try:
        # Crear mensaje
        msg = MIMEMultipart('alternative')
        msg['From'] = email_from
        msg['To'] = ', '.join(email_to)
        msg['Subject'] = f"{subject_prefix} {alert.severity.upper()}: {alert.component} - {alert.metric}"
        
        # Crear versión HTML
        html_content = format_email_html(alert)
        html_part = MIMEText(html_content, 'html', 'utf-8')
        msg.attach(html_part)
        
        # Enviar email
        with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as server:
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.send_message(msg)
        
        logger.info(f"✅ Email enviado a {', '.join(email_to)}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error enviando email: {e}", exc_info=True)
        return False


# ============ Orquestador Principal ============

def send_notification(alert: Alert, channels: Optional[List[str]] = None) -> Dict[str, bool]:
    """
    Orquestador principal para enviar notificaciones.
    
    Args:
        alert: Alert a enviar
        channels: Lista de canales (None = usar configuración)
    
    Returns:
        Dict con resultados por canal: {'slack': True, 'email': False}
    """
    if not NOTIFICATIONS_ENABLED:
        logger.debug("Notificaciones deshabilitadas")
        return {}
    
    # Verificar rate limiting
    if not check_rate_limit(alert):
        logger.info(f"⏸️  Alerta en rate limit: {alert.component}/{alert.metric}")
        return {}
    
    # Determinar canales
    if channels is None:
        channels = [ch.strip() for ch in NOTIFICATION_CHANNELS]
    
    results = {}
    
    # Enviar a cada canal
    for channel in channels:
        channel = channel.lower().strip()
        
        try:
            if channel == 'slack':
                results['slack'] = send_slack_notification(alert)
            elif channel == 'email':
                results['email'] = send_email_notification(alert)
            else:
                logger.warning(f"⚠️ Canal desconocido: {channel}")
                results[channel] = False
                
        except Exception as e:
            logger.error(f"❌ Error enviando a {channel}: {e}", exc_info=True)
            results[channel] = False
    
    # Registrar envío
    if any(results.values()):
        record_alert_sent(alert, results)
    
    # Si falla, guardar en pending_alerts
    if not any(results.values()):
        try:
            save_pending_alert(alert, channels)
        except Exception as e:
            logger.error(f"❌ Error guardando alerta pendiente: {e}", exc_info=True)
    
    return results


def save_pending_alert(alert: Alert, channels: List[str]) -> None:
    """Guarda alerta no enviada para reintento posterior"""
    try:
        postgres = get_postgres()
        session = postgres.get_session()
        
        insert_sql = """
        INSERT INTO pending_alerts (
            alert_data,
            channels,
            next_retry_at
        ) VALUES (
            :alert_data,
            :channels,
            :next_retry_at
        )
        """
        
        alert_dict = {
            'severity': alert.severity,
            'component': alert.component,
            'metric': alert.metric,
            'message': alert.message,
            'current_value': alert.current_value,
            'previous_value': alert.previous_value,
            'threshold': alert.threshold,
            'recommendation': alert.recommendation,
            'created_at': alert.created_at.isoformat()
        }
        
        next_retry = datetime.now() + timedelta(hours=1)
        
        session.execute(
            text(insert_sql),
            {
                'alert_data': json.dumps(alert_dict),
                'channels': json.dumps(channels),
                'next_retry_at': next_retry
            }
        )
        session.commit()
        session.close()
        
        logger.info(f"💾 Alerta guardada para reintento: {alert.component}/{alert.metric}")
        
    except Exception as e:
        logger.error(f"❌ Error guardando alerta pendiente: {e}", exc_info=True)


def send_batch_notifications(alerts: List[Alert]) -> Dict[str, Dict]:
    """
    Envía notificaciones en batch con digest.
    
    Args:
        alerts: Lista de alertas
    
    Returns:
        Dict con estadísticas de envío
    """
    # Separar por severidad
    critical_alerts = [a for a in alerts if a.severity == 'critical']
    warning_alerts = [a for a in alerts if a.severity == 'warning']
    
    stats = {
        'critical_sent': 0,
        'warning_sent': 0,
        'failed': 0
    }
    
    # Enviar críticas inmediatamente
    for alert in critical_alerts:
        results = send_notification(alert)
        if any(results.values()):
            stats['critical_sent'] += 1
        else:
            stats['failed'] += 1
    
    # Para warnings, enviar digest si hay >10
    if len(warning_alerts) > 10:
        # TODO: Implementar digest de warnings
        logger.info(f"📊 {len(warning_alerts)} alertas warning (enviar digest)")
        stats['warning_sent'] = len(warning_alerts)
    else:
        # Enviar individualmente
        for alert in warning_alerts:
            results = send_notification(alert)
            if any(results.values()):
                stats['warning_sent'] += 1
            else:
                stats['failed'] += 1
    
    return stats

