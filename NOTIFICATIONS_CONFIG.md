# Configuración de Notificaciones

## Variables de Entorno

Agregar al archivo `.env`:

```bash
# Notificaciones generales
NOTIFICATIONS_ENABLED=true
NOTIFICATION_CHANNELS=slack,email  # Separados por coma

# Slack
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
SLACK_CHANNEL=#chatbot-alerts
SLACK_USERNAME=Chatbot Monitor
SLACK_ICON_EMOJI=:robot_face:

# Email (SMTP)
EMAIL_ENABLED=true
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
EMAIL_FROM=chatbot-alerts@yourcompany.com
EMAIL_TO=team@yourcompany.com,devops@yourcompany.com
EMAIL_SUBJECT_PREFIX=[CHATBOT ALERT]
```

## Configuración de Slack

1. Crear un webhook en Slack:
   - Ir a https://api.slack.com/apps
   - Crear nueva app o usar existente
   - Ir a "Incoming Webhooks"
   - Activar webhooks
   - Agregar nuevo webhook al workspace
   - Copiar la URL del webhook

2. Configurar en `.env`:
   ```bash
   SLACK_WEBHOOK_URL=https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX
   SLACK_CHANNEL=#chatbot-alerts
   ```

## Configuración de Email (Gmail)

1. Habilitar "App Passwords" en Gmail:
   - Ir a https://myaccount.google.com/apppasswords
   - Generar nueva contraseña de aplicación
   - Usar esta contraseña (no la contraseña normal)

2. Configurar en `.env`:
   ```bash
   EMAIL_ENABLED=true
   SMTP_HOST=smtp.gmail.com
   SMTP_PORT=587
   SMTP_USERNAME=your-email@gmail.com
   SMTP_PASSWORD=your-16-char-app-password
   EMAIL_FROM=your-email@gmail.com
   EMAIL_TO=team@yourcompany.com
   ```

## Testing

### Test Slack
```bash
python scripts/test_notifications.py --channel slack --severity critical
```

### Test Email
```bash
python scripts/test_notifications.py --channel email --severity warning
```

### Test Todos los Canales
```bash
python scripts/test_notifications.py --all
```

## Comportamiento por Severidad

- **CRITICAL**: Enviar inmediatamente a todos los canales habilitados
- **WARNING**: Agregar a digest, enviar cada 6 horas (si hay >10)
- **INFO**: Solo logging, no enviar notificaciones

## Rate Limiting

- Máximo 5 alertas críticas por hora por componente
- Cooldown de 1 hora entre alertas similares
- Alertas en cooldown se registran pero no se envían

## Dashboard

Ver historial de notificaciones:

```bash
GET /metrics/notifications/history
```

Retorna:
- Total enviadas
- Por canal (slack, email)
- Por severidad (critical, warning)
- Fallos de entrega
- Últimas 24 horas

## Manejo de Errores

- Si Slack falla, intentar Email
- Si ambos fallan, guardar en `pending_alerts` para reintento
- Reintentar `pending_alerts` cada hora
- No romper el flujo principal si fallan notificaciones

## Troubleshooting

### Slack no envía notificaciones

1. Verificar que `SLACK_WEBHOOK_URL` es válida
2. Probar webhook manualmente:
   ```bash
   curl -X POST -H 'Content-type: application/json' \
     --data '{"text":"Test"}' \
     YOUR_WEBHOOK_URL
   ```
3. Verificar que el canal existe y el bot tiene permisos

### Email no envía notificaciones

1. Verificar que `EMAIL_ENABLED=true`
2. Verificar credenciales SMTP
3. Para Gmail, usar "App Password" no contraseña normal
4. Verificar que el puerto 587 no está bloqueado
5. Probar conexión SMTP:
   ```python
   import smtplib
   server = smtplib.SMTP('smtp.gmail.com', 587)
   server.starttls()
   server.login('your-email@gmail.com', 'app-password')
   ```

### Notificaciones no se envían

1. Verificar `NOTIFICATIONS_ENABLED=true`
2. Verificar `NOTIFICATION_CHANNELS` incluye los canales deseados
3. Verificar rate limiting (puede estar en cooldown)
4. Revisar logs: `docker-compose logs app | grep notification`

