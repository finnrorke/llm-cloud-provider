module "networking" {
  source = "../../modules/networking"

  org_prefix                        = var.org_prefix
  environment                       = var.environment
  vpc_cidr                          = var.vpc_cidr
  availability_zone                 = var.availability_zone
  public_subnet_cidrs               = var.public_subnet_cidrs
  private_subnet_cidrs              = var.private_subnet_cidrs
  tags                              = var.tags
  kubernetes_api_allowed_cidrs      = var.kubernetes_api_allowed_cidrs
  kubernetes_nodeport_allowed_cidrs = var.kubernetes_nodeport_allowed_cidrs

  stack_name  = local.stack_name
  common_tags = local.common_tags
}

module "llm" {
  source = "../../modules/llm"

  stack_name                                 = local.stack_name
  common_tags                                = local.common_tags
  vpc_id                                     = module.networking.vpc_id
  availability_zone                          = var.availability_zone
  control_plane_subnet_id                    = module.networking.public_subnet_ids[0]
  cpu_infra_subnet_id                        = module.networking.private_subnet_ids[0]
  kubernetes_control_plane_security_group_id = module.networking.kubernetes_control_plane_security_group_id
  kubernetes_node_security_group_id          = module.networking.kubernetes_node_security_group_id
  control_plane_instance_type                = var.control_plane_instance_type
  cpu_infra_instance_type                    = var.cpu_infra_instance_type
  gpu_test_instance_count                    = var.gpu_test_instance_count
  gpu_test_instance_type                     = var.gpu_test_instance_type
  cluster_token                              = var.cluster_token
  cluster_cidr                               = var.cluster_cidr
  service_cidr                               = var.service_cidr
}
