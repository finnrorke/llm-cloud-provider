variable "region" {
  type        = string
  description = "AWS region for the environment."
  default     = "us-east-1"
}

variable "org_prefix" {
  type        = string
  description = "Short prefix used in AWS resource names."
}

variable "environment" {
  type        = string
  description = "Environment name used for tagging and naming."
  default     = "dev"
}

variable "project_name" {
  type        = string
  default     = "llm-cloud-provider"
  description = "Project name used for tagging"
}

variable "vpc_cidr" {
  type        = string
  description = "CIDR block for the VPC."
  default     = "10.42.0.0/16"
}

variable "availability_zone" {
  type        = string
  description = "Availability zone used for the VPC subnets and EC2 instances."
  default     = "eu-west-2a"
}

variable "public_subnet_cidrs" {
  type        = list(string)
  description = "CIDR blocks for the public subnets."
  //default     = ["10.42.1.0/24", "10.42.2.0/24", "10.42.3.0/24"]
  default = ["10.42.1.0/24"]
}

variable "private_subnet_cidrs" {
  type        = list(string)
  description = "CIDR blocks for private subnets"
  default     = ["10.42.2.0/24"]
  //default     = ["10.42.4.0/24", "10.42.5.0/24", "10.42.6.0/24"]
}

variable "tags" {
  type        = map(string)
  description = "Additional tags to apply to all resources."
  default     = {}
}

variable "kubernetes_api_allowed_cidrs" {
  type        = list(string)
  description = "CIDR blocks allowed to reach the Kubernetes API server on port 6443."
  default     = []
}

variable "kubernetes_nodeport_allowed_cidrs" {
  type        = list(string)
  description = "CIDR blocks allowed to reach Kubernetes NodePort services on ports 30000-32767."
  default     = []
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
  default     = 3
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
