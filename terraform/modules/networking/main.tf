module "vpc" {
  source = "terraform-aws-modules/vpc/aws"

  name            = "${var.stack_name}-vpc"
  cidr            = var.vpc_cidr
  azs             = [var.availability_zone]
  private_subnets = var.private_subnet_cidrs
  public_subnets  = var.public_subnet_cidrs

  tags = var.common_tags
}

module "nat_instance" {
  source = "int128/nat-instance/aws"

  name                        = "main"
  vpc_id                      = module.vpc.vpc_id
  public_subnet               = module.vpc.public_subnets[0]
  private_subnets_cidr_blocks = module.vpc.private_subnets_cidr_blocks
  private_route_table_ids     = module.vpc.private_route_table_ids
}
