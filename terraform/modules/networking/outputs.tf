output "vpc_id" {
  description = "VPC ID created for the environment."
  value       = module.vpc.vpc_id
}

output "public_subnet_ids" {
  description = "Public subnet IDs created for the environment."
  value       = module.vpc.public_subnets
}

output "private_subnet_ids" {
  description = "Private subnet IDs created for the environment."
  value       = module.vpc.private_subnets
}

output "kubernetes_control_plane_security_group_id" {
  description = "Security group ID for Kubernetes control plane nodes."
  value       = aws_security_group.kubernetes_control_plane.id
}

output "kubernetes_node_security_group_id" {
  description = "Security group ID for Kubernetes worker nodes."
  value       = aws_security_group.kubernetes_nodes.id
}
