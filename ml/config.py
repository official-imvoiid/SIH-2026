"""Central configuration. Import from here; never hardcode a path or a seed."""

from pathlib import Path

RANDOM_SEED = 42

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "ml" / "artifacts"

# Elliptic ships 49 timesteps. Split temporally, never randomly: a random split leaks
# future information and inflates illicit F1 by roughly ten points.
TRAIN_TS_MAX = 34
VAL_TS_MAX = 39
# Timestep 43 is where a real dark-market shutdown changes the illicit population and
# published models collapse. We report metrics either side of it deliberately.
DRIFT_TS = 43
N_TIMESTEPS = 49

# Server-side cap on returned subgraph size; see docs/ARCHITECTURE.md.
MAX_SUBGRAPH_NODES = 300
DEFAULT_HOPS = 2

MODEL_VERSION = "0.1.0-baseline"
