import paramiko
hostname = 'connect.westb.seetacloud.com'
port = 41178
username = 'root'
password = '4WnmfYzE1oJA'
cmd = "nohup bash -c 'apt-get update && apt-get install -y git-lfs && git lfs install && source /etc/network_turbo && rm -rf Cat_Hunter && git clone https://github.com/dunja-g/Cat_Hunter.git && cd Cat_Hunter && git lfs pull && unzip -o deploy_package.zip && pip install pandas scikit-learn matplotlib seaborn torch torchvision opencv-python && python src/preprocess.py && echo \"Starting Phase 1\" && python src/train.py --model transfer --epochs 10 --no_aug && echo \"Starting Phase 2\" && python src/train.py --model transfer --epochs 30 --fine_tune --weights models/best_transfer.pth' > /root/train.log 2>&1 &"
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(hostname, port=port, username=username, password=password, timeout=10)
stdin, stdout, stderr = ssh.exec_command(cmd)
exit_status = stdout.channel.recv_exit_status()
print(f"Exit code: {exit_status}")
ssh.close()
