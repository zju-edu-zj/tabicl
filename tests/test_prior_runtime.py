"""Native runtime startup, dry runs and explicit compatibility overrides."""
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest
import torch

from tabicl.prior.graph_lib._config import PriorConfig
from tabicl.train._train_config import build_parser

ROOT = Path(__file__).resolve().parents[1]


def module():
    spec = importlib.util.spec_from_file_location('prepare_prior_runtime', ROOT/'scripts/prepare_prior_runtime.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_identity_tracks_source_and_torch(tmp_path):
    m = module()
    source = tmp_path/'op.cpp'
    source.write_text('a')
    first = m.runtime_identity(source, torch)
    source.write_text('b')
    second = m.runtime_identity(source, torch)
    assert first['source_sha256'] != second['source_sha256']
    assert first['torch'] == torch.__version__
    assert first['cxx11_abi'] == bool(torch._C._GLIBCXX_USE_CXX11_ABI)


def test_python_cli_and_shell_fast_prior_defaults(tmp_path):
    assert PriorConfig().categorical_sampling == 'marginal'
    assert PriorConfig().graph_execution == 'optimized'
    assert PriorConfig().filter_execution == 'sufficient'
    args = build_parser().parse_args([])
    assert (args.categorical_sampling, args.graph_execution, args.filter_execution,
            args.dataset_filter_mode) == ('marginal', 'optimized', 'sufficient', 'full')
    env = dict(os.environ, DRY_RUN='1', CKPT_DIR=str(tmp_path), LOG_DIR=str(tmp_path),
               PYTHON='/fake/python-must-not-run')
    for key in ('PRIOR_CATEGORICAL_SAMPLING', 'PRIOR_GRAPH_EXECUTION', 'PRIOR_FILTER_EXECUTION',
                'PRIOR_BACKEND', 'PRIOR_SCHEDULE'):
        env.pop(key, None)
    text = subprocess.check_output(['bash', str(ROOT/'scripts/train_v2_clf_stage1.sh')], env=env, text=True)
    tokens = shlex.split(text)
    assert tokens[:3] == ['/fake/python-must-not-run', '-m', 'torch.distributed.run']
    args = build_parser().parse_args(tokens[tokens.index('--wandb_log'):])
    assert PriorConfig.from_args(args).graph_execution == 'optimized'
    assert args.categorical_sampling == 'marginal'
    assert (args.prior_backend, args.prior_schedule) == ('node_pool', 'ready_micro')


def test_fast_wrapper_resolves_precision_and_interpreter_without_building(tmp_path):
    env = dict(os.environ, DRY_RUN='1', CKPT_DIR=str(tmp_path), LOG_DIR=str(tmp_path),
               PYTHON='/fake/python-must-not-run', PPU_SDK_SETUP=str(tmp_path/'no-sdk'))
    env.pop('TRAIN_DTYPE', None)
    text = subprocess.check_output(['bash', str(ROOT/'scripts/train_v2_clf_stage1_fast.sh')], env=env, text=True)
    tokens = shlex.split(text)
    assert tokens[:3] == ['/fake/python-must-not-run', '-m', 'torch.distributed.run']
    assert tokens[tokens.index('--dtype')+1] == 'bfloat16'


def test_explicit_prior_compatibility_overrides(tmp_path):
    env = dict(os.environ, DRY_RUN='1', CKPT_DIR=str(tmp_path), LOG_DIR=str(tmp_path),
        PRIOR_GRAPH_EXECUTION='reference', PRIOR_CATEGORICAL_SAMPLING='legacy',
        PRIOR_FILTER_EXECUTION='reference', PYTHON='/fake/python')
    text = subprocess.check_output(['bash', str(ROOT/'scripts/train_v2_clf_stage1.sh')], env=env, text=True)
    tokens = shlex.split(text)
    args = build_parser().parse_args(tokens[tokens.index('--wandb_log'):])
    assert (args.graph_execution, args.categorical_sampling, args.filter_execution) == (
        'reference', 'legacy', 'reference')


def test_fast_wrapper_adds_selected_python_console_tools_to_path(tmp_path):
    scripts = tmp_path/'scripts'
    scripts.mkdir()
    wrapper = scripts/'train_v2_clf_stage1_fast.sh'
    wrapper.write_text((ROOT/'scripts/train_v2_clf_stage1_fast.sh').read_text())
    (scripts/'train_v2_clf_stage1.sh').write_text('printf \'%s\\n\' "$PATH"\n')
    interpreter = tmp_path/'environment with spaces'/'bin'/'python'
    env = dict(os.environ, PYTHON=str(interpreter), PPU_SDK_SETUP=str(tmp_path/'no-sdk'))
    result = subprocess.check_output(['bash', str(wrapper)], env=env, text=True)
    assert result.strip().split(os.pathsep)[0] == str(interpreter.parent)


def test_prepare_explicit_library_stdout_is_one_path():
    library = os.environ.get('TABICL_PRIOR_DISTANCE_LIBRARY')
    if not library:
        pytest.skip('native library not configured')
    result = subprocess.run([sys.executable, str(ROOT/'scripts/prepare_prior_runtime.py'),
        '--library', library], text=True, capture_output=True, check=True)
    assert result.stdout.strip() == str(Path(library).resolve())
    assert json.loads(result.stderr.splitlines()[-1])['prior_library'] == result.stdout.strip()


def test_prepare_cold_build_and_reuse_stdout_are_one_path(tmp_path):
    outputs = []
    for _ in range(2):
        result = subprocess.run([sys.executable, str(ROOT/'scripts/prepare_prior_runtime.py'),
            '--cache', str(tmp_path/'native cache')], text=True, capture_output=True, check=True)
        library = Path(result.stdout.strip())
        assert library.is_file() and library.suffix == '.so'
        assert result.stdout == str(library)+'\n'
        assert json.loads(result.stderr.splitlines()[-1])['prior_library'] == str(library)
        assert (library.parent/'runtime.json').is_file()
        outputs.append(library)
    assert outputs[0] == outputs[1]


def test_prepare_missing_library_fails_before_workers(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT/'scripts/prepare_prior_runtime.py'),
        '--library', str(tmp_path/'missing.so')], text=True, capture_output=True)
    assert result.returncode != 0 and not result.stdout.strip()


def test_benchmark_launch_command_uses_resolved_interpreter(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('benchmark_training_schedule', ROOT/'scripts/benchmark_training_schedule.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(sys, 'argv', ['benchmark', '--steps', '1', '--output', str(tmp_path)])
    dry_text = subprocess.check_output(['bash', str(ROOT/'scripts/train_v2_clf_stage1.sh')],
        cwd=ROOT, text=True, env=dict(os.environ, DRY_RUN='1', PYTHON='/fake/vendor/python',
            CKPT_DIR=str(tmp_path/'ckpt'), LOG_DIR=str(tmp_path/'logs')))
    monkeypatch.setattr(mod.subprocess, 'check_output', lambda *a, **k: dry_text)
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return type('Result', (), {'returncode': 0})()
    monkeypatch.setattr(mod.subprocess, 'run', run)
    with pytest.raises(SystemExit) as error:
        mod.main()
    assert error.value.code == 0
    assert calls[0][:4] == [sys.executable, '-m', 'torch.distributed.run', '--standalone']
    assert '/fake/vendor/python' not in calls[0]
