import sys
from pathlib import Path


def pytest_configure(config):
    project_dir = str(Path(__file__).resolve().parent)
    sys.path.insert(0, project_dir)
