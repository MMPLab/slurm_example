#!/usr/bin/env python
import hydra.utils as hydra_utils
import hydra
import submitit
import logging
import copy
from pathlib import Path
import warnings
import random
import os
import numpy as np

os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

MAIN_PID = os.getpid()
SIGNAL_RECEIVED = False

log = logging.getLogger(__name__)


def update_pythonpath_relative_hydra():
    """Update PYTHONPATH to only have absolute paths."""
    # NOTE: We do not change sys.path: we want to update paths for future instantiations
    # of python using the current environment (namely, when submitit loads the job
    # pickle).
    try:
        original_cwd = Path(hydra_utils.get_original_cwd()).resolve()
    except (AttributeError, ValueError):
        # Assume hydra is not initialized, we don't need to do anything.
        # In hydra 0.11, this returns AttributeError; later it will return ValueError
        # https://github.com/facebookresearch/hydra/issues/496
        # I don't know how else to reliably check whether Hydra is initialized.
        return
    paths = []
    for orig_path in os.environ["PYTHONPATH"].split(":"):
        path = Path(orig_path)
        if not path.is_absolute():
            path = original_cwd / path
        paths.append(path.resolve())
    os.environ["PYTHONPATH"] = ":".join([str(x) for x in paths])
    log.info('PYTHONPATH: {}'.format(os.environ["PYTHONPATH"]))


class Worker:
    def __call__(self, args):
        import torch.multiprocessing as mp
        import importlib
        import numpy as np

        mp.set_start_method('spawn')
        main_function = getattr(importlib.import_module(args.worker), 'main')
        args = copy.deepcopy(args)

        np.set_printoptions(precision=3)
        socket_name = os.popen(
            "ip r | grep default | awk '{print $5}'").read().strip('\n')
        print("Setting GLOO and NCCL sockets IFNAME to: {}".format(socket_name))
        os.environ["GLOO_SOCKET_IFNAME"] = socket_name

        job_env = submitit.JobEnvironment()
        args.env.rank = job_env.global_rank

        # Use random port to avoid collision between parallel processes
        if args.env.world_size == 1:
            args.env.port = np.random.randint(10000, 20000)
        args.env.dist_url = f'tcp://{job_env.hostnames[0]}:{args.env.port}'
        print('Using url {}'.format(args.env.dist_url))

        if args.seed == -1:
            args.seed = None

        if args.env.gpu is not None:
            warnings.warn(
                'You have chosen a specific GPU. This will completely '
                'disable data parallelism.')

        # Run code
        main_function(args)

    # Reques unfinished jobs
    def checkpoint(self, *args, **kwargs) -> submitit.helpers.DelayedSubmission:
        return submitit.helpers.DelayedSubmission(self, *args, **kwargs)


def my_jobs():
    return os.popen('squeue -o %j -u $USER').read().split("\n")


@hydra.main(config_path='./configs', config_name='train_cifar', version_base='1.1')
def main(args):
    update_pythonpath_relative_hydra()
    args.output = hydra_utils.to_absolute_path(args.output)
    args.log.ckpt_dir = hydra_utils.to_absolute_path(args.log.ckpt_dir)

    # If job is running, ignore
    job_names = my_jobs()
    if args.env.slurm and args.exp_name in job_names:
        print(f'Skipping {args.exp_name} because already in queue')
        return

    # If model is trained, ignore
    ckpt_fname = os.path.join(args.log.ckpt_dir, args.exp_name,
                              'checkpoint_{:04d}.pth')
    if os.path.exists(ckpt_fname.format(args.epochs - 1)):
        print(f'Skipping {args.exp_name} because already finished training')
        return

    # Submit jobs
    executor = submitit.AutoExecutor(
        folder=args.log.submitit_dir,
        slurm_max_num_timeout=100,
        cluster=None if args.env.slurm else "debug",
    )
    additional_parameters = {}
    if args.env.nodelist != "":
        additional_parameters.update({"nodelist": args.env.nodelist})
    if args.env.exclude != "":
        additional_parameters.update({"exclude": args.env.exclude})
    executor.update_parameters(
        timeout_min=args.env.slurm_timeout,
        slurm_partition=args.env.slurm_partition,
        cpus_per_task=args.env.workers,
        gpus_per_node=args.env.ngpu,
        nodes=args.env.world_size,
        tasks_per_node=1,
        mem_gb=args.env.mem_gb,
        slurm_additional_parameters=additional_parameters,
        slurm_signal_delay_s=120)
    executor.update_parameters(name=args.exp_name)
    job = executor.submit(Worker(), args)
    if not args.env.slurm:
        job.result()


if __name__ == '__main__':
    main()
