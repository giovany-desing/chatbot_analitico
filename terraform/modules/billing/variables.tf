variable "project_name" {
  description = "Nombre del proyecto"
  type        = string
}

variable "environment" {
  description = "Ambiente (prod, dev, etc)"
  type        = string
}

variable "monthly_budget_limit" {
  description = "Límite de presupuesto mensual en USD"
  type        = number
  default     = 25
}

variable "alert_email" {
  description = "Email para recibir alertas de presupuesto"
  type        = string
}

