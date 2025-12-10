output "budget_name" {
  description = "Nombre del presupuesto creado"
  value       = aws_budgets_budget.monthly_cost.name
}

output "budget_id" {
  description = "ID del presupuesto"
  value       = aws_budgets_budget.monthly_cost.id
}

