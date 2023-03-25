# Language Control Diffusion

<div align="center">

[![Build status](https://github.com/ezhang7423/language-control-diffusion/workflows/build/badge.svg?branch=master&event=push)](https://github.com/ezhang7423/language-control-diffusion/actions?query=workflow%3Abuild)
[![Dependencies Status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)](https://github.com/ezhang7423/language-control-diffusion/pulls?utf8=%E2%9C%93&q=is%3Apr%20author%3Aapp%2Fdependabot)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Security: bandit](https://img.shields.io/badge/security-bandit-green.svg)](https://github.com/PyCQA/bandit)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/ezhang7423/language-control-diffusion/blob/master/.pre-commit-config.yaml)
[![Semantic Versions](https://img.shields.io/badge/%20%20%F0%9F%93%A6%F0%9F%9A%80-semantic--versions-e10079.svg)](https://github.com/ezhang7423/language-control-diffusion/releases)
[![License](https://img.shields.io/github/license/ezhang7423/language-control-diffusion)](https://github.com/ezhang7423/language-control-diffusion/blob/master/LICENSE)

Efficiently scaling through space, time, and tasks

</div>

## 🚀 Features

### Fast, efficient and easy training

- We've spent months optimizing this code to make diffusion models train fast in CALVIN, and can now offer fully training to 300K gradient steps in **8 hours**, and evaluation on 1000 tasks in **6 hours**. We get these numbers on the A10G 24GB.

### Flexible policy abstraction levels

- You can easily adjust the amount of burden to put on the high level vs the low level policy by simply adjusting the temporal stride, or the `clip_stride` as referred to in the code

### Universal interface

- Our high level diffusion policy is independent of the low level policy. This means you can **plug and play** different low level policies and environments.

### SOTA performance on MT-LHC CALVIN

- We achieve an average 88.7% success on the horizon length 1 SR of the multitask long-horizon control problem in CALVIN. This is the highest performance yet of any model that incorporates no inductive biases on the problem.

### Cached CALVIN Datasets

- We offer a daemon script that runs in the background, caching the entire shared memory dataset of HULC in the background to save you ~20 minutes at the start of each training run, which aggregates to hours and possibly days over the course of a full research project

### Development features

- Supports for `Python 3.8` and higher.
- Simple and flexible development interface for fast extensibility to new environments and models
- [`Poetry`](https://python-poetry.org/) as the dependencies manager. See configuration in [`pyproject.toml`](https://github.com/ezhang7423/language-control-diffusion/blob/master/pyproject.toml) and [`setup.cfg`](https://github.com/ezhang7423/language-control-diffusion/blob/master/setup.cfg).
- Automatic codestyle with [`black`](https://github.com/psf/black), [`isort`](https://github.com/timothycrosley/isort) and [`pyupgrade`](https://github.com/asottile/pyupgrade).
- Ready-to-use [`pre-commit`](https://pre-commit.com/) hooks with code-formatting.
- Type checks with [`mypy`](https://mypy.readthedocs.io); docstring checks with [`darglint`](https://github.com/terrencepreilly/darglint); security checks with [`safety`](https://github.com/pyupio/safety) and [`bandit`](https://github.com/PyCQA/bandit)
- Testing with [`pytest`](https://docs.pytest.org/en/latest/).
- Ready-to-use [`.editorconfig`](https://github.com/ezhang7423/language-control-diffusion/blob/master/.editorconfig), [`.dockerignore`](https://github.com/ezhang7423/language-control-diffusion/blob/master/.dockerignore), and [`.gitignore`](https://github.com/ezhang7423/language-control-diffusion/blob/master/.gitignore). You don't have to worry about those things.

## 🧪 Prior Experiment Logs

An example of a recent LCD training run for your reference can be found [here](https://wandb.ai/lang-diffusion/vanilla-diffuser/reports/LCD-training-run--VmlldzozODgyNzE4?accessToken=i8ccudz5b89j51wiktp2dl7sqothdo54iyq4wyd9ldk5g25joi5yvov0pc25vw5h). We will also be releasing the entire experiment logs of all experiments done for this project soon.

## ⚓ Installation

We require either Mambaforge or Conda and [git lfs](https://anaconda.org/conda-forge/git-lfs). We highly recommend [Mambaforge](https://mamba.readthedocs.io/en/latest/installation.html) (a faster drop-in replacement of conda). [Conda](https://docs.conda.io/en/latest/miniconda.html) is also supported. To install, simply run

```bash
$ make install
```

This will set up a new conda environment `lcd`. It will also download all necessary data (~9.5 GB/), including

- HULC seeds
- On-policy offline datasets
- LCD seeds

### Options

If you would like to install just the repository without downloading data, run

```bash
$ make NO_DATA=1 install
```

If you just want to download the data, run

```bash
$ git clone https://github.com/ezhang7423/hulc-data.git --recurse-submodules
```


### Troubleshooting

We have thoroughly tested the installation process and environment with NVIDIA GPUs CUDA Versions 11.6, 11.7, and 12.0 on Ubuntu 18.04, 20.04, and AlmaLinux 8.7 (similar to RHEL). Running on windows or macOS will likely present difficulties. If any part of the installation fails, you can activate the environment with `mamba/conda activate lcd` and try debugging what the issue is by running individual commands in the `install` section of the `Makefile`.

If you see this error:

```
UnsatisfiableError: The following specifications were found
to be incompatible with the existing python installation in your environment:

Specifications:

  - pytorch3d=0.7.2 -> python[version='>=2.7,<2.8.0a0|>=3.11,<3.12.0a0|>=3.7,<3.8.0a0|>=3.6,<3.7.0a0|>=3.5,<3.6.0a0']

Your python: python=3.8
```

try using mamba instead of conda.

### Usage

Once the repository is installed, all scripts can be run with `lcd`. This is also equivalent to directly running `python ./src/lcd/__main__.py`, which may be preferable when using a debugger.

```
> lcd                                                          (lcd) 
                                                                     
 Usage: lcd [OPTIONS] COMMAND [ARGS]...                              
                                                                     
╭─ Options ─────────────────────────────────────────────────────────╮
│ --install-completion          Install completion for the current  │
│                               shell.                              │
│ --show-completion             Show completion for the current     │
│                               shell, to copy it or customize the  │
│                               installation.                       │
│ --help                        Show this message and exit.         │
╰───────────────────────────────────────────────────────────────────╯
╭─ Commands ────────────────────────────────────────────────────────╮
│ rollout     Rollout in the environment for evaluation or dataset  │
│             collection                                            │
│ train_hulc  Train the original hulc model                         │
│ train_lcd   Train the original hulc model                         │
╰───────────────────────────────────────────────────────────────────╯
```
Train HULC

`lcd train_hulc` 

Evaluate HULC

`lcd rollout hulc`

Generate on-policy data

`lcd rollout generate`

Train LCD

`lcd train_lcd`

Evaluate LCD

`lcd rollout lcd`

It's really that easy! If you would like to pass more training options to hulc or diffuser, both can be done in the exact same way as in the original repositories (see here for [hulc](https://github.com/ezhang7423/hulc-baseline#training) and here for [diffuser](https://github.com/jannerm/diffuser#training-from-scratch)). These scripts will simply pass the arguments to the underlying original code. It should be noted that the `dataset` key is no longer supported for diffuser as there is only one dataset, and that the `task_D_D` dataset needs to be downloaded first by following the directions at `./language-control-diffusion/submodules/hulc-baseline/dataset`.


If using wandb, please change the key `wanb_entity` in `./lcd/config/calvin.py` and `./submodules/hulc-baseline/conf/logger/wandb.yaml` to your team or username.

## 📈 Releases

You can see the list of available releases on the [GitHub Releases](https://github.com/ezhang7423/language-control-diffusion/releases) page.

We follow [Semantic Versions](https://semver.org/) specification, and use the [`Release Drafter`](https://github.com/marketplace/actions/release-drafter). As pull requests are merged, a draft release is kept up-to-date listing the changes, ready to publish when you’re ready. With the categories option, you can categorize pull requests in release notes using labels.

### List of labels and corresponding titles

|               **Label**               |  **Title in Releases**  |
| :-----------------------------------: | :---------------------: |
|       `enhancement`, `feature`        |       🚀 Features       |
| `bug`, `refactoring`, `bugfix`, `fix` | 🔧 Fixes & Refactoring  |
|       `build`, `ci`, `testing`        | 📦 Build System & CI/CD |
|              `breaking`               |   💥 Breaking Changes   |
|            `documentation`            |    📝 Documentation     |
|            `dependencies`             | ⬆️ Dependencies updates |

GitHub creates the `bug`, `enhancement`, and `documentation` labels for you. Dependabot creates the `dependencies` label. Create the remaining labels on the Issues tab of your GitHub repository, when you need them.

## 🏗️ Development

### Directory Structure

```
.
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
├── Makefile
├── README.md
├── SECURITY.md
├── cookiecutter-config-file.yml
├── poetry.lock
├── pyproject.toml
├── requirements.txt
├── setup.cfg
├── submodules
│   ├── hulc-baseline # original hulc code
│   └── hulc-data
│       ├── (indices,ordering,seq).pt # evaluation sequences sorted by task, useful for single task evaluation
│       ├── README.md
│       ├── annotations.json # the training and evaluation language descriptions of all tasks
│       ├── default_1000_sequences.pt # the default 1000 tasks that are generated during evaluation. this is cached to speed up eval by ~3 min per run.
│       ├── hulc-baselines-30 # a collection of 3 seeds of HULC, each run for 30 epochs trained in the same manner with the original code checkpointed at hulc-baseline
│       ├── hulc-trajectories # the offline on-policy datasets used to train lcd. these are saved in the latent space of the LLP, after processing through its encoder
│       ├── lcd-seeds # high level diffusion policies trained on the aforementioned seeds 
│       └── t5-v1_1-xxl_embeddings.pt # t5 XXL embeddings of all annotations
└── src
    └── lcd
        ├── __init__.py
        ├── __main__.py # the primary entrypoint. delegates commands by automatically parsing the apps directory
        ├── __pycache__
        ├── apps # this holds all the ways of interacting with this repo
        │   ├── __pycache__
        │   ├── rollout.py # rollout allows you to evaluate hulc, lcd, and collect on-policy datasets
        │   ├── train_hulc.py # a wrapper around the original hulc training code in hulc-baseline.
        │   └── train_lcd.py # a wrapper around the diffuser training code, which calls ./src/lcd/apps/diffuser.py
        ├── config
        │   ├── __pycache__
        │   └── calvin.py # holds all configuration for training diffuser on the calvin benchmark
        ├── datasets
        │   ├── __init__.py
        │   ├── __pycache__
        │   └── sequence.py # load the prior mentioned on-policy datasets
        ├── models
        │   ├── __init__.py
        │   ├── __pycache__
        │   ├── diffusion.py
        │   ├── helpers.py
        │   └── temporal.py
        ├── scripts # older entrypoints that will be transitioned to apps
        │   ├── diffuser.py # training and evaluation of diffuser
        │   └── generation
        │       ├── dataset.py # generate the on-policy dataset from the goal space
        │       ├── embeddings.py # generate the t5 XXL embeddings
        │       └── task_orderings.py # generate an updated version of the (indices,ordering,seq).pt file in ./submodules/hulc-data
        └── utils
            ├── __init__.py
            ├── __pycache__
            ├── arrays.py
            ├── config.py
            ├── eval.py # the meat of the rollout and evaluation code
            ├── git_utils.py
            ├── serialization.py
            ├── setup.py
            ├── timer.py
            └── training.py # the meat of the diffuser training code
```

1. Install and initialize poetry and install `pre-commit` hooks:

```bash
make install
make pre-commit-install
```

2. Run the codestyle:

```bash
make codestyle
```

### Makefile usage

[`Makefile`](https://github.com/ezhang7423/language-control-diffusion/blob/master/Makefile) contains a lot of functions for faster development.

<details>
<summary>1. Download and remove Poetry</summary>
<p>

To download and install Poetry run:

```bash
make poetry-download
```

To uninstall

```bash
make poetry-remove
```

</p>
</details>

<details>
<summary>2. Install all dependencies and pre-commit hooks</summary>
<p>

Install requirements:

```bash
make install
```

Pre-commit hooks coulb be installed after `git init` via

```bash
make pre-commit-install
```

</p>
</details>

<details>
<summary>3. Codestyle</summary>
<p>

Automatic formatting uses `pyupgrade`, `isort` and `black`.

```bash
make codestyle

# or use synonym
make formatting
```

Codestyle checks only, without rewriting files:

```bash
make check-codestyle
```

> Note: `check-codestyle` uses `isort`, `black` and `darglint` library

Update all dev libraries to the latest version using one comand

```bash
make update-dev-deps
```

<details>
<summary>4. Code security</summary>
<p>

```bash
make check-safety
```

This command launches `Poetry` integrity checks as well as identifies security issues with `Safety` and `Bandit`.

```bash
make check-safety
```

</p>
</details>

</p>
</details>

<details>
<summary>5. Type checks</summary>
<p>

Run `mypy` static type checker

```bash
make mypy
```

</p>
</details>

<details>
<summary>6. Tests with coverage badges</summary>
<p>

Run `pytest`

```bash
make test
```

</p>
</details>

<details>
<summary>7. All linters</summary>
<p>

Of course there is a command to ~~rule~~ run all linters in one:

```bash
make lint
```

the same as:

```bash
make test && make check-codestyle && make mypy && make check-safety
```

</p>
</details>

<details>
<summary>8. Docker</summary>
<p>

```bash
make docker-build
```

which is equivalent to:

```bash
make docker-build VERSION=latest
```

Remove docker image with

```bash
make docker-remove
```

More information [about docker](https://github.com/ezhang7423/language-control-diffusion/tree/master/docker).

</p>
</details>

<details>
<summary>9. Cleanup</summary>
<p>
Delete pycache files

```bash
make pycache-remove
```

Remove package build

```bash
make build-remove
```

Delete .DS_STORE files

```bash
make dsstore-remove
```

Remove .mypycache

```bash
make mypycache-remove
```

Or to remove all above run:

```bash
make cleanup
```

</p>
</details>

### Poetry

Want to know more about Poetry? Check [its documentation](https://python-poetry.org/docs/).

<details>
<summary>Details about Poetry</summary>
<p>

Poetry's [commands](https://python-poetry.org/docs/cli/#commands) are very intuitive and easy to learn, like:

- `poetry add numpy@latest`
- `poetry run pytest`
- `poetry publish --build`

etc

</p>
</details>

<!-- ### Building and releasing

Building a new version of the application contains steps:

- Bump the version of your package `poetry version <version>`. You can pass the new version explicitly, or a rule such as `major`, `minor`, or `patch`. For more details, refer to the [Semantic Versions](https://semver.org/) standard.
- Make a commit to `GitHub`.
- Create a `GitHub release`.
- And... publish 🙂 `poetry publish --build` -->

## 🎯 What's next

Replanning, further probing the language generalization, trying discrete state-action spaces, scaling the model are all interesting research directions to explore. As far as further improvements to this repository go, we are planning to add

1. Parallel training (multiple runs, one per gpu)
2. Diffuser from pixels
3. BC-Z and RT-1 ablations

## 🛡 License

[![License](https://img.shields.io/github/license/ezhang7423/language-control-diffusion)](https://github.com/ezhang7423/language-control-diffusion/blob/master/LICENSE)

This project is licensed under the terms of the `MIT` license. See [LICENSE](https://github.com/ezhang7423/language-control-diffusion/blob/master/LICENSE) for more details.

## 📃 Citation

```bibtex
@article{language-control-diffusion,
  author={Zhang, Edwin and Lu, Yujie and Wang, William and Zhang, Amy},
  title={LAD: Language Control Diffusion: efficiently scaling through Space, Time, and Tasks},
  year = {2023},
  journal={arXiv preprint arXiv:2210.15629},
  howpublished = {\url{https://github.com/ezhang7423/language-control-diffusion}}
}
```

# 👏 Credits [![🚀 Your next Python package needs a bleeding-edge project structure.](https://img.shields.io/badge/python--package--template-%F0%9F%9A%80-brightgreen)](https://github.com/TezRomacH/python-package-template)

Massive thanks to Oier Mees and Luka Shermann for providing te [CALVIN](https://github.com/mees/calvin) and [HULC](https://github.com/lukashermann/hulc) codebases, and huge thanks to Michael Janner and Yilun Du for providing the [Diffuser](https://github.com/jannerm/diffuser) codebase. This work would not be possible without standing on the shoulders of these giants.

Template: [`python-package-template`](https://github.com/TezRomacH/python-package-template)
