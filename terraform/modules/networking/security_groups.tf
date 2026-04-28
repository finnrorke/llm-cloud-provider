// Kubernetes control plane
resource "aws_security_group" "kubernetes_control_plane" {
  name        = "${var.stack_name}-kubernetes-control-plane"
  description = "Security group for Kubernetes control plane nodes."
  vpc_id      = module.vpc.vpc_id

  tags = merge(
    var.common_tags,
    {
      Name                                      = "${var.stack_name}-kubernetes-control-plane"
      Role                                      = "kubernetes-control-plane"
      "kubernetes.io/cluster/${var.stack_name}" = "owned"
    },
  )
}

//Rules
resource "aws_vpc_security_group_egress_rule" "kubernetes_control_plane_all" {
  security_group_id = aws_security_group.kubernetes_control_plane.id
  cidr_ipv4         = "0.0.0.0/0"
  ip_protocol       = "-1"
  description       = "Allow all outbound traffic from the Kubernetes control plane."
}

resource "aws_vpc_security_group_ingress_rule" "kubernetes_api_from_nodes" {
  security_group_id            = aws_security_group.kubernetes_control_plane.id
  referenced_security_group_id = aws_security_group.kubernetes_nodes.id
  ip_protocol                  = "tcp"
  from_port                    = 6443
  to_port                      = 6443
  description                  = "Allow worker nodes to reach the Kubernetes API server."
}

resource "aws_vpc_security_group_ingress_rule" "kubernetes_api_from_allowed_cidrs" {
  for_each          = toset(var.kubernetes_api_allowed_cidrs)
  security_group_id = aws_security_group.kubernetes_control_plane.id
  cidr_ipv4         = each.value
  ip_protocol       = "tcp"
  from_port         = 6443
  to_port           = 6443
  description       = "Allow approved CIDRs to reach the Kubernetes API server."
}

resource "aws_vpc_security_group_ingress_rule" "kubernetes_control_plane_etcd" {
  security_group_id            = aws_security_group.kubernetes_control_plane.id
  referenced_security_group_id = aws_security_group.kubernetes_control_plane.id
  ip_protocol                  = "tcp"
  from_port                    = 2379
  to_port                      = 2380
  description                  = "Allow etcd traffic between control plane nodes."
}

resource "aws_vpc_security_group_ingress_rule" "kubernetes_control_plane_components" {
  security_group_id            = aws_security_group.kubernetes_control_plane.id
  referenced_security_group_id = aws_security_group.kubernetes_control_plane.id
  ip_protocol                  = "tcp"
  from_port                    = 10257
  to_port                      = 10259
  description                  = "Allow control plane components to communicate with each other."
}

resource "aws_vpc_security_group_ingress_rule" "kubernetes_control_plane_from_nodes_all" {
  security_group_id            = aws_security_group.kubernetes_control_plane.id
  referenced_security_group_id = aws_security_group.kubernetes_nodes.id
  ip_protocol                  = "-1"
  description                  = "Allow the k3s server node to participate in node-to-node and pod overlay traffic."
}

//Kubernetes Nodes
resource "aws_security_group" "kubernetes_nodes" {
  name        = "${var.stack_name}-kubernetes-nodes"
  description = "Security group for Kubernetes worker nodes."
  vpc_id      = module.vpc.vpc_id

  tags = merge(
    var.common_tags,
    {
      Name                                      = "${var.stack_name}-kubernetes-nodes"
      Role                                      = "kubernetes-nodes"
      "kubernetes.io/cluster/${var.stack_name}" = "owned"
    },
  )
}

//Rules
resource "aws_vpc_security_group_egress_rule" "kubernetes_nodes_all" {
  security_group_id = aws_security_group.kubernetes_nodes.id
  cidr_ipv4         = "0.0.0.0/0"
  ip_protocol       = "-1"
  description       = "Allow all outbound traffic from Kubernetes worker nodes."
}

resource "aws_vpc_security_group_ingress_rule" "kubernetes_nodes_self" {
  security_group_id            = aws_security_group.kubernetes_nodes.id
  referenced_security_group_id = aws_security_group.kubernetes_nodes.id
  ip_protocol                  = "-1"
  description                  = "Allow node-to-node and pod overlay traffic between Kubernetes workers."
}

resource "aws_vpc_security_group_ingress_rule" "kubernetes_nodes_from_control_plane_all" {
  security_group_id            = aws_security_group.kubernetes_nodes.id
  referenced_security_group_id = aws_security_group.kubernetes_control_plane.id
  ip_protocol                  = "-1"
  description                  = "Allow the k3s control plane node to participate in node-to-node and pod overlay traffic."
}

resource "aws_vpc_security_group_ingress_rule" "kubernetes_kubelet_from_control_plane" {
  security_group_id            = aws_security_group.kubernetes_nodes.id
  referenced_security_group_id = aws_security_group.kubernetes_control_plane.id
  ip_protocol                  = "tcp"
  from_port                    = 10250
  to_port                      = 10250
  description                  = "Allow the control plane to talk to kubelets on worker nodes."
}

resource "aws_vpc_security_group_ingress_rule" "kubernetes_nodeport_from_allowed_cidrs" {
  for_each          = toset(var.kubernetes_nodeport_allowed_cidrs)
  security_group_id = aws_security_group.kubernetes_nodes.id
  cidr_ipv4         = each.value
  ip_protocol       = "tcp"
  from_port         = 30000
  to_port           = 32767
  description       = "Allow approved CIDRs to reach Kubernetes NodePort services."
}
