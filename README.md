# Instructions

### Set up conda environment 
Make sure the cuda version is right for the pytorch command.
```
conda create -n base-env python=3.7 scikit-learn scipy opencv numpy -y
conda activate base-env
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -y
conda install -c anaconda pillow -y
conda install -c conda-forge submitit -y
pip install hydra-core --upgrade
pip install tensorboard
```

### Training example:
Model is saved under `${output}/${exp_name}/checkpoints`.
Logs are saved under `${output}/${exp_name}/logs`.
```bash
PYTHONPATH=. python main_slurm.py \
  output=/grogu/user/pmorgado/workspace/slurm_example/checkpoints \
  exp_name=r18pt_cifar10_bs\${batch_size}_lr\${lr} \
  arch=resnet18 pretrained=true \
  epochs=100 lr=0.1 batch_size=1024 \
  env.slurm=True env.distributed=True env.world_size=1 env.ngpu=4 env.workers=10 env.slurm_partition="abhinav\,all"
```

### Hyper-parameter search (on batch_size in this case):
```bash
PYTHONPATH=. python main_slurm.py \
  output=/grogu/user/pmorgado/workspace/slurm_example/checkpoints \
  exp_name=r18pt_cifar10_bs\${batch_size}_lr\${lr} \
  arch=resnet18 pretrained=true \
  epochs=100 lr=0.1 batch_size=256,1024,4096 \
  env.slurm=True env.distributed=True env.world_size=1 env.ngpu=4 env.workers=10 env.slurm_partition="abhinav\,all"
```

### Test:
Loads `${output}/${exp_name}/checkpoints/checkpoint_latest.pth` and evaluates it.
```bash
PYTHONPATH=. python main_slurm.py \
  output=/grogu/user/pmorgado/workspace/slurm_example/checkpoints \
  exp_name=r18pt_cifar10_bs\${batch_size}_lr\${lr} \
  arch=resnet18 batch_size=256 evaluate=True \
  env.slurm=True env.distributed=True env.world_size=1 env.ngpu=4 env.workers=10 env.slurm_partition="abhinav\,all"
```

### Training without slurm:
First get into an interactive job:
```bash
srun -N 1 -t 6:00:00 -G 4 --cpus-per-task=10 --mem=128000  --partition=abhinav,all  --pty /bin/bash
```

Then, run the job with `slurm=False`:
```bash
PYTHONPATH=. python main_slurm.py \
  output=/grogu/user/pmorgado/workspace/slurm_example/checkpoints \
  exp_name=r18pt_cifar10_bs\${batch_size}_lr\${lr} \
  arch=resnet18 pretrained=true \
  epochs=100 lr=0.1 batch_size=1024 \
  env.slurm=False env.distributed=True env.workers=10
  ```