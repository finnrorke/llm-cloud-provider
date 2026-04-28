output "vpc_id" {
  description = "VPC ID created for the environment."
  value       = module.networking.vpc_id
}

output "kubernetes_control_plane_security_group_id" {
  description = "Security group ID for Kubernetes control plane nodes."
  value       = module.networking.kubernetes_control_plane_security_group_id
}

output "kubernetes_node_security_group_id" {
  description = "Security group ID for Kubernetes worker nodes."
  value       = module.networking.kubernetes_node_security_group_id
}
