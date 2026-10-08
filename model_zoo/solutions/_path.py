"""Lets the solution scripts import the shared modules (zoo, common, runtimes, checks) one folder up."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
