"""Build ONCE before starting workers; never compile in a training worker."""
import argparse
from pathlib import Path
import torch
from torch.utils.cpp_extension import load

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--build-directory", type=Path, required=True)
args = parser.parse_args()
args.build_directory.mkdir(parents=True, exist_ok=True)
source = Path(__file__).resolve().parents[1] / "src/tabicl/prior/graph_lib/_nearest_cpu.cpp"
library = load(name="tabicl_prior_exact_cpu", sources=[str(source)],
               extra_cflags=["-O3", "-fno-fast-math", "-ffp-contract=off"],
               build_directory=str(args.build_directory), is_python_module=False,
               verbose=True)
print("TABICL_PRIOR_DISTANCE_LIBRARY=" + str(library))
