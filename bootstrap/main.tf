resource "aws_s3_bucket" "aws_terraform_state_bucket" {
  bucket = "${var.org_prefix}-terraform-state"
}

resource "aws_dynamodb_table" "aws_terraform_lock_db" {
  name = "${var.org_prefix}-terraform-state-lock"
  hash_key = "LockID"
  attribute {
    name = "LockID"
    type = "S"
  }
}
