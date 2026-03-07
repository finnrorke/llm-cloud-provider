locals {
  stack_name = "${var.org_prefix}-${var.environment}-${var.name}"

  common_tags = merge(
    {
      ManagedBy   = "terraform"
      Project     = "llm-cloud-provider"
      Environment = var.environment
      Service     = var.name
    },
    var.tags,
  )
}
