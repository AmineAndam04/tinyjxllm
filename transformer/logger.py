import contextlib
import json
import os
import subprocess


class Logger:
    """
    A logger. It support Tensorboard and W&B
    """

    def __init__(self, args, run_name):
        self.backend = args.logger

        self.log_dir = os.path.join(args.save_path, run_name)
        os.makedirs(self.log_dir, exist_ok=True)

        config = vars(args).copy()
        with contextlib.suppress(subprocess.CalledProcessError, OSError):
            config["git_commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                text=True,
            ).strip()

        if self.backend == "tensorboard":
            from tensorboardX import SummaryWriter

            self.writer = SummaryWriter(self.log_dir)
            self.writer.add_text("config", json.dumps(config, indent=4))

        elif self.backend == "wandb":
            import wandb

            self.run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                config=config,
                dir=self.log_dir,
            )

        else:
            raise ValueError(f"Unknown logging backend: {self.backend}")

        with open(os.path.join(self.log_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

    def log(self, name, value, step):
        value = float(value)

        if self.backend == "tensorboard":
            self.writer.add_scalar(name, value, step)
        else:
            self.run.log({name: value}, step=step, commit=False)

    def close(self, exit_code=0):
        if self.backend == "tensorboard":
            self.writer.close()
        else:
            self.run.finish(exit_code=exit_code)
