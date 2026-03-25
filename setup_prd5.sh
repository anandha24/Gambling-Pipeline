#!/bin/bash
set -e

echo "[1] Install Supervisor"
source ~/.bashrc
sudo apt update
sudo apt install -y supervisor

echo "[2] Create Supervisor config (prd5_api)"

sudo tee /etc/supervisor/conf.d/prd5_api.conf > /dev/null <<'EOF'
[program:prd5_api]
command=/workspace/Gambling-Pipeline/miniconda3/bin/conda run -n prd5 python run_server.py
directory=/workspace/Gambling-Pipeline
autostart=true
autorestart=true
startsecs=5
stderr_logfile=/workspace/logs/prd5.err.log
stdout_logfile=/workspace/logs/prd5.out.log
EOF

echo "[3] Reload Supervisor"
sudo /usr/bin/supervisord -c /etc/supervisor/supervisord.conf
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl restart prd5_api

echo "[4] Start prd5_api"
sudo supervisorctl start prd5_api || sudo supervisorctl restart prd5_api

echo "Done!"
sudo supervisorctl status