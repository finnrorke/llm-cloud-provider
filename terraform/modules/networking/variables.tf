variable "org_prefix" {
  type        = string
  description = "Short prefix used in AWS resource names."
}

variable "environment" {
  type        = string
  description = "Environment name used for tagging and naming."
}

variable "vpc_cidr" {
  type        = string
  description = "CIDR block for the VPC."
  default     = "10.42.0.0/16"
}

variable "availability_zone" {
  type        = string
  description = "Availability zone used for the VPC subnets."
}

variable "public_subnet_cidrs" {
  type        = list(string)
  description = "CIDR blocks for the public subnets."
  default     = ["10.42.1.0/24", "10.42.2.0/24", "10.42.3.0/24"]
}

variable "private_subnet_cidrs" {
  type        = list(string)
  description = "CIDR blocks for private subnets"
  default     = ["10.42.4.0/24", "10.42.5.0/24", "10.42.6.0/24"]
}

variable "tags" {
  type        = map(string)
  description = "Additional tags to apply to all resources."
  default     = {}
}

variable "stack_name" {
  type        = string
  description = "Stack name tag for resource identification"
}

variable "common_tags" {
  type        = map(string)
  description = "Universal tags"
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
