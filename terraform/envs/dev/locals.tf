locals {
  stack_name = "${var.org_prefix}-${var.environment}-${var.project_name}"

  common_tags = merge(
    {
      ManagedBy   = "terraform"
      Project     = var.project_name
      Environment = var.environment
    },
    var.tags,
  )
}
