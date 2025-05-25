# sta141c-final-proj

# Prerequisites
[Conda](https://www.anaconda.com/download/success)

# Installation process
1. `conda create -n sta141c-final python=3.12`
2. `conda activate sta141c-final`
3. `pip install poetry`
4. `poetry install`

# Configure Environment on VSCode
1. Enter `Cmd+Shift+p` on Mac or `Ctrl+Shift+p` on PC
2. Type `Python: select interpreter`
3. Select the Anaconda `sta141c-final` env

# Adding a dependency
`poetry add [pip package name]`

# Removing a dependency
`poetry remove [pip package name]`

# Pulling updates
* Run `git pull`
* Run `poetry install`