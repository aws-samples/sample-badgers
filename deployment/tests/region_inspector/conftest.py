"""Put the region_inspector Lambda source on sys.path so tests import the real modules.

The modules under test (region_inspector, comparison, blind_reader) import only Pillow / re /
json, so they load without the foundation layer or boto3. lambda_handler is intentionally not
imported here — it depends on the foundation layer and AWS.
"""
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[2] / "lambdas" / "code" / "region_inspector"
sys.path.insert(0, str(CODE_DIR))
