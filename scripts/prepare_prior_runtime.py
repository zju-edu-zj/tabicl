"""Build and validate the full CPU prior runtime once, before torchrun.

Print only the resolved library path to stdout (suitable for shell capture).
Compiler output and the runtime manifest go to stderr. Never replace the
installed accelerator PyTorch package. Artifacts live inside the repository.
"""
import argparse
from contextlib import contextmanager, redirect_stdout
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import sysconfig


REQUIRED_OPS = ('nearest_cpu', 'nearest_vector_cpu', 'cdist_vector_cpu', 'tree_cpu',
                'quadratic_features_cpu', 'em_probabilities_cpu',
                'compact_rows_cpu', 'column_std_mean_cpu')


@contextmanager
def compiler_output_to_stderr():
    # Ninja inherits fd 1 directly: redirect_stdout alone only redirects Python
    # writes and contaminates shell-captured paths on a cold build.
    sys.stdout.flush()
    saved_stdout = os.dup(1)
    try:
        os.dup2(2, 1)
        with redirect_stdout(sys.stderr):
            yield
    finally:
        sys.stderr.flush()
        os.dup2(saved_stdout, 1)
        os.close(saved_stdout)


def runtime_identity(source, torch):
    return dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        torch=torch.__version__, torch_git=torch.version.git_version,
        cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        python=sysconfig.get_config_var('SOABI'), platform=platform.platform(),
        machine=platform.machine(), capability=torch.backends.cpu.get_cpu_capability(),
        compiler=os.environ.get('CXX', 'c++'), flags_version=1)


def validate_library(path, torch):
    torch.ops.load_library(str(path))
    missing = [op for op in REQUIRED_OPS if not hasattr(torch.ops.tabicl_prior_exact, op)]
    if missing:
        raise RuntimeError(f'{path}: missing operators {missing}; unset '
            'TABICL_PRIOR_DISTANCE_LIBRARY to build the current source')
    # Run real instructions in this process, before any training workers exist.
    x = torch.tensor([[1., 2.], [3., 6.]])
    stats = torch.ops.tabicl_prior_exact.column_std_mean_cpu(x)
    torch.testing.assert_close(stats, torch.tensor([[2., 4.], [1., 2.]]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library', type=Path)
    p.add_argument('--cache', type=Path)
    p.add_argument('--check-training', action='store_true')
    a = p.parse_args()
    import torch
    import sklearn
    root = Path(__file__).resolve().parents[1]
    source = root/'src/tabicl/prior/graph_lib/_nearest_cpu.cpp'
    if a.check_training:
        for module in ('numpy', 'scipy', 'joblib', 'threadpoolctl', 'einops', 'psutil',
                       'tqdm', 'huggingface_hub', 'transformers', 'wandb', 'xgboost'):
            try:
                importlib.import_module(module)
            except ImportError as error:
                raise RuntimeError(f'Missing training dependency {module}; see '
                    'requirements-prior-performance.txt (keep vendor torch installed)') from error
        if not torch.cuda.is_available():
            raise RuntimeError('Accelerator unavailable; activate the CUDA/PPU SDK before launching')
    identity = runtime_identity(source, torch)
    if a.library is not None:
        library = a.library.expanduser().resolve(strict=True)
        validate_library(library, torch)
    else:
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
        build = (a.cache or root/'.cache/prior-native')/key
        build.mkdir(parents=True, exist_ok=True)
        # cpp_extension's build lock serializes concurrent launchers. Each source
        # / torch / CPU identity has a separate directory; no stale .so reuse.
        if not shutil.which(os.environ.get('CXX', 'c++')):
            raise RuntimeError('C++ compiler unavailable; install g++ (Debian: apt-get install build-essential)')
        if not shutil.which('ninja'):
            raise RuntimeError('ninja unavailable; run python -m pip install ninja')
        from torch.utils.cpp_extension import load
        flags = ['-O3', '-fno-fast-math', '-ffp-contract=off']
        if identity['capability'] in ('AVX2', 'AVX512'):
            flags += ['-DCPU_CAPABILITY_AVX2', '-mavx2', '-mfma', '-mf16c']
        os.environ.setdefault('MAX_JOBS', '1')
        with compiler_output_to_stderr():
            library = Path(load(name='tabicl_prior_exact_cpu', sources=[str(source)],
                extra_cflags=flags, build_directory=str(build), is_python_module=False,
                verbose=True)).resolve()
        # load() already registered the operators; don't load a second binary.
        validate_library(library, torch)
        (build/'runtime.json').write_text(json.dumps(identity | dict(library=str(library),
            sklearn=sklearn.__version__, interpreter=sys.executable), indent=2)+'\n')
    print(json.dumps(dict(prior_library=str(library), torch=torch.__version__,
        sklearn=sklearn.__version__, interpreter=sys.executable), sort_keys=True), file=sys.stderr)
    print(library)


if __name__ == '__main__':
    main()
