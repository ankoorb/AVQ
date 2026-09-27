```bash
# Laptop
zip -r AVQ.zip AVQ -x "*.DS_Store" -x "*__pycache__*" -x "*.git*" 

# Runpod Setup
unzip AVQ.zip -d /workspace/
chmod +x /workspace/AVQ/setup_env.sh
/workspace/AVQ/setup_env.sh

# Train
cd /workspace/AVQ
python -m compileall avq
python -m avq.train --model_seed 2027 --experiment_dir /workspace/places_2027 --experiment_id 11

# Output for download
tar -czvf avq_32x32_k1024_d64.tar.gz --exclude='*epoch_*.pt' avq_32x32_k1024_d64/
```