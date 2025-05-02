#!/bin/bash
uv sync
uv pip install -e .
source .venv/bin/activate

cd example/Lorenz96
python make_data.py