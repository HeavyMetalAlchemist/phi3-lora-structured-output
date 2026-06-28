# outputs.tf

output "instance_id" {
  description = "EC2 Instance ID"
  value       = aws_instance.g4dn.id
}

output "public_ip" {
  description = "Public IP address"
  value       = aws_instance.g4dn.public_ip
}

output "ssh_command" {
  description = "SSH command to connect"
  value       = "ssh -i ~/.ssh/test-llm.pem ec2-user@${aws_instance.g4dn.public_ip}"
}