# Cognitive Robotics Coursework

## Requirements

- `Python 3` (developed using `Python 3.7.7`)

## Setup

1. Create a Python virtual environment (e.g. `conda create --name cognitive python=3.7.7` or `virtualenv env`)
2. Activate the environment (e.g. `conda activate nls` or `source ./env/bin/activate`)
3. Install dependencies
   - Using [Poetry](https://github.com/python-poetry/poetry) (recommended)
      - Deployment: `poetry install --no-dev`
      - Development: `poetry install`
   - Using Pip (only for deployment): `pip install -r requirements.txt`
4. Run the program from the root of the project: `python src/main.py`

## Project structure

- `src` - All the code written for the courseworks
  - `main.py` - The script to run to evaluate classifier
  - `resnet.py` - Definition of the ResNet model
