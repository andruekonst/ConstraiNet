from argparse import ArgumentParser
from pathlib import Path
import shutil
import logging
from datetime import datetime
from benedict import benedict
from .run import RunConfig, RunExecutor
from ..utils import get_git_revision_hash


def load_config(config_file: str) -> RunConfig:
    config = benedict(config_file)
    return RunConfig(**config)


def cli():
    # logging.basicConfig(level=logging.INFO)

    parser = ArgumentParser('Run constrainted optimization experiments')
    parser.add_argument('config', type=str, help='Experiment configuration file')
    args = parser.parse_args()

    config_file = args.config
    results_path = Path(config_file).resolve().parent / 'results'
    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_path = results_path / f'{now}'
    assert not results_path.exists()
    results_path.mkdir(parents=True, exist_ok=False)

    # new logging
    logging.basicConfig(
        filename=str(results_path / 'log.txt'),
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.DEBUG
    )
    logging.getLogger().addHandler(logging.StreamHandler())

    # get Git info
    git_hash = get_git_revision_hash()
    logging.info(f'GIT_HASH={git_hash}')

    # copy config
    out_config_path = results_path / 'used_config.yaml'
    shutil.copy(config_file, out_config_path)
    with out_config_path.open('a') as out_conf:
        out_conf.write("\n")
        out_conf.write(f"# git_hash: {git_hash}  # dumped at {now}")
        out_conf.write("\n")

    runner = RunExecutor(config=load_config(config_file), path=results_path)
    runner.run()


if __name__ == '__main__':
    cli()
