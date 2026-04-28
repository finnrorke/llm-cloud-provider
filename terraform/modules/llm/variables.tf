variable "stack_name" {
  type        = string
  description = "Stack name tag for resource identification."
}

variable "common_tags" {
  type        = map(string)
  description = "Universal tags applied to the LLM instances."
}

variable "vpc_id" {
  type        = string
  description = "VPC ID where the LLM instances will be created."
}

variable "availability_zone" {
  type        = string
  description = "Availability zone where the LLM instances will run."
}

variable "control_plane_subnet_id" {
  type        = string
  description = "Subnet ID used by the Kubernetes control plane instance."
}

variable "cpu_infra_subnet_id" {
  type        = string
  description = "Subnet ID used by the CPU infrastructure instance."
}

variable "kubernetes_control_plane_security_group_id" {
  type        = string
  description = "Security group ID for the Kubernetes control plane instance."
}

variable "kubernetes_node_security_group_id" {
  type        = string
  description = "Security group ID for the Kubernetes CPU infrastructure instance."
}

variable "control_plane_instance_type" {
  type        = string
  description = "EC2 instance type for the Kubernetes control plane."
  default     = "t3.medium"
}

variable "cpu_infra_instance_type" {
  type        = string
  description = "EC2 instance type for the CPU infrastructure node."
  default     = "t3.large"
}

variable "gpu_test_instance_count" {
  type        = number
  description = "Number of GPU-backed Kubernetes test nodes to create."
  default     = 2
}

variable "gpu_test_instance_type" {
  type        = string
  description = "EC2 instance type for the GPU-backed Kubernetes test nodes."
  default     = "g4dn.xlarge"
}

variable "cluster_token" {
  type        = string
  description = "Shared secret used by k3s servers and agents to form the cluster."
  sensitive   = true
}

variable "cluster_cidr" {
  type        = string
  description = "Pod network CIDR for the k3s cluster."
  default     = "10.44.0.0/16"
}

variable "service_cidr" {
  type        = string
  description = "Service CIDR for the k3s cluster."
  default     = "10.43.0.0/16"
}
