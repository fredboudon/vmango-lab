# vmango-lab

A simulation and analysis tool for mango tree architecture.

## Install

```
conda env create -f environment.yml
python setup.py develop
```

## From **zero** to V-Mango on windows 10

### Download

- miniconda https://docs.conda.io/en/latest/miniconda.html
- git https://git-scm.com/download/win
- vscode https://code.visualstudio.com/download

### Install

- Run all installers (if you are unsure just use the default options)
- Open vscode and open the default windows terminal availabel in vscode
- run the following commands from the terminal one after the other:

```
git clone https://github.com/jvail/vmango-lab.git
cd vmango-lab
conda env create -f environment.yml
conda activate vmango-lab
conda install -c conda-forge nodejs
python setup.py develop
```

### Run Jupyter

If all goes well you can run Jupyter from the vscode terminal (with the `vmango-lab` conda environment being active)

```
jupyter lab
```

You will find several notebooks in the vmango-lab/notebooks
