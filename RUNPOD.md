# RunPod Deployment Guide

This guide covers deploying and training the transformer model on RunPod Spot instances.

## Overview

The codebase includes built-in support for RunPod Spot instances:
- **Graceful shutdown** on SIGTERM (spot preemption)
- **Automatic checkpointing** to network volume
- **Resume capability** from any checkpoint
- **Path resolution** to `/runpod-volume`

## Quick Start

```bash
# On RunPod instance
cd /runpod-volume
git clone <your-repo>
cd TransformersFromScratch

pip install -r requirements.txt

# Start training
python train.py --config configs/experiment_config_runpod.yaml
```

## Configuration

### Local Development
Use the default config (RunPod features disabled):
```bash
python train.py --config configs/experiment_config.yaml
```

### RunPod Deployment
Use the RunPod-specific config:
```bash
python train.py --config configs/experiment_config_runpod.yaml
```

### Configuration Options

```yaml
# configs/experiment_config_runpod.yaml
runpod:
  enabled: true                          # Enable RunPod mode
  base_path: "/runpod-volume"            # Network volume mount point
  emergency_checkpoint_on_signal: true   # Save on SIGTERM
  checkpoint_every_n_steps: 100          # Frequent saves for spot resilience
  auto_resume: true                      # Auto-find latest checkpoint
```

## Storage Layout

When RunPod mode is enabled, all paths are prefixed with `/runpod-volume`:

```
/runpod-volume/
├── checkpoints/
│   └── {timestamp}/
│       ├── epoch_0.pt
│       ├── epoch_1.pt
│       ├── last_state.pt          # Most recent state (for resume)
│       └── emergency_*.pt         # Saved on preemption
├── logs/
│   └── {experiment_name}/
│       └── {timestamp}/
│           └── events.out.tfevents.*
└── models/
    └── tokenizers/
        └── europarl_tokenizer.json
```

## Common Patterns

### Fresh Training
```bash
python train.py --config configs/experiment_config_runpod.yaml
```

### Resume After Preemption
```bash
# Auto-find latest checkpoint
python train.py --config configs/experiment_config_runpod.yaml --resume

# Or specify checkpoint explicitly
python train.py --config configs/experiment_config_runpod.yaml \
    --checkpoint /runpod-volume/checkpoints/20240115_143022/last_state.pt
```

### Run Training in Background
```bash
# Using nohup (survives SSH disconnect)
nohup python train.py --config configs/experiment_config_runpod.yaml > train.log 2>&1 &

# Check progress
tail -f train.log
```

### Run with Screen (Recommended)
```bash
# Start screen session
screen -S training

# Run training
python train.py --config configs/experiment_config_runpod.yaml

# Detach: Ctrl+A, then D
# Reattach later: screen -r training
```

## TensorBoard

### Method 1: RunPod HTTP Proxy
```bash
# On RunPod (run in background)
tensorboard --logdir=/runpod-volume/logs --port=6006 --host=0.0.0.0 &
```
Access via RunPod dashboard → Connect → HTTP Service [6006]

### Method 2: SSH Tunnel (More Stable)
```bash
# On local machine
ssh -L 6006:localhost:6006 root@<POD_IP> -p <SSH_PORT> -i ~/.ssh/runpod_key

# On RunPod
tensorboard --logdir=/runpod-volume/logs --port=6006 &
```
Access via `http://localhost:6006`

## Pod Setup Checklist

### 1. Create Network Volume
- Navigate to **Storage** in RunPod console
- Create volume (50GB recommended) in your target data center
- Name it (e.g., `transformer-storage`)

### 2. Deploy Pod
- Select same data center as network volume
- Choose GPU (RTX 4090 recommended)
- Select **Spot** instance for cost savings
- Template: `runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04`
- Container disk: 20GB
- Mount network volume to `/runpod-volume`
- Expose ports: 22 (TCP), 6006 (HTTP)

### 3. First-Time Setup
```bash
# SSH into pod
ssh root@<POD_IP> -p <SSH_PORT> -i ~/.ssh/runpod_key

# Clone repo to network volume
cd /runpod-volume
git clone <your-repo>
cd TransformersFromScratch

# Install dependencies
pip install -r requirements.txt

# Verify GPU
nvidia-smi
```

## Handling Preemption

When a Spot instance is preempted:

1. RunPod sends `SIGTERM` to the container
2. Training loop detects signal via `GracefulKiller`
3. Emergency checkpoint saved to `/runpod-volume/checkpoints/*/last_state.pt`
4. Process exits cleanly

### After Preemption
1. Deploy a new pod in the same data center
2. Mount the same network volume
3. Resume training:
   ```bash
   python train.py --config configs/experiment_config_runpod.yaml --resume
   ```

## Monitoring

### Check Training Progress
```bash
# View recent log output
tail -f train.log

# Check GPU usage
watch -n 1 nvidia-smi

# List checkpoints
ls -la /runpod-volume/checkpoints/*/
```

### Check Disk Usage
```bash
df -h /runpod-volume
```

## Downloading Results

### Using SCP
```bash
# From local machine
scp -P <SSH_PORT> -i ~/.ssh/runpod_key \
    root@<POD_IP>:/runpod-volume/checkpoints/*/epoch_*.pt ./
```

### Using runpodctl
```bash
# Install runpodctl locally
# Then download files
runpodctl receive <POD_ID>:/runpod-volume/checkpoints/
```

## Troubleshooting

### Training Won't Resume
```bash
# Check if checkpoints exist
ls -la /runpod-volume/checkpoints/

# Try specifying checkpoint explicitly
python train.py --resume --checkpoint /runpod-volume/checkpoints/<timestamp>/last_state.pt
```

### Out of Disk Space
```bash
# Check usage
df -h

# Clean old checkpoints (keep last 3 epochs)
cd /runpod-volume/checkpoints/<timestamp>
ls -t epoch_*.pt | tail -n +4 | xargs rm -f
```

### GPU Memory Error
Reduce batch size in config:
```yaml
data:
  batch_size: 32  # Reduced from 64
```

### TensorBoard Not Loading
```bash
# Check if running
ps aux | grep tensorboard

# Restart with correct binding
pkill tensorboard
tensorboard --logdir=/runpod-volume/logs --port=6006 --host=0.0.0.0 &
```

## Cost Optimization

1. **Use Spot instances** - Up to 80% cheaper than on-demand
2. **Checkpoint frequently** - `checkpoint_every_n_steps: 100`
3. **Stop pod when idle** - Don't leave pods running overnight
4. **Use appropriate GPU** - RTX 4090 offers best price/performance for training
5. **Clean old checkpoints** - Network volume storage has monthly costs
