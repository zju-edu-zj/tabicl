import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location('prior_trace_analysis',
    Path(__file__).resolve().parents[1]/'scripts/analyze_prior_trace.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_waits_use_union_and_only_due_waits_attribute_blockers(tmp_path):
    records = [dict(event='complete', task_id=7, started_monotonic_s=4.,
        encoded_monotonic_s=8., generated_monotonic_s=7., received_s=9., submitted_s=0.),
        dict(event='dispatch', step=1, at_s=9., task_ids=[7]),
        dict(event='wait', step=1, reason='due_micro', started_s=2., finished_s=6., missing_task_ids=[7]),
        dict(event='wait', step=1, reason='due_micro', started_s=3., finished_s=7., missing_task_ids=[7]),
        dict(event='wait', step=0, reason='insufficient_ready', started_s=0., finished_s=1., missing_task_ids=[7])]
    (tmp_path/'service.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    result = analysis.analyze(tmp_path)
    assert result['wait_union_s'] == dict(due_micro=5., insufficient_ready=1.)
    assert result['due_wait_with_queued_blocker_union_s'] == 2.
    assert result['due_wait_with_running_blocker_union_s'] == 3.
    assert result['queue_s']['p50'] == 4.
    assert result['transit_s']['p50'] == 1.
    assert result['due_blockers'][0]['rank_wait_s'] == 8.  # diagnostic rank sum only
    result = analysis.analyze(tmp_path, skip_steps=1)
    assert 'insufficient_ready' not in result['wait_union_s']
    assert result['dispatched_steps'] == 1


def test_interval_union_handles_nesting_overlap_and_empty():
    assert analysis.union_seconds([]) == 0
    assert analysis.union_seconds([(0, 4), (1, 2), (3, 7), (9, 10)]) == pytest.approx(8)


def test_transit_is_distinct_from_running_and_post_receive_wakeup(tmp_path):
    records = [dict(event='complete', task_id=7, started_monotonic_s=4.,
        encoded_monotonic_s=8., generated_monotonic_s=7., received_s=9., submitted_s=0.),
        dict(event='wait', step=1, reason='due_micro', started_s=7., finished_s=10., missing_task_ids=[7])]
    (tmp_path/'service.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    result = analysis.analyze(tmp_path)
    assert result['wait_union_s']['due_micro'] == 3.
    assert result['due_wait_with_queued_blocker_union_s'] == 0.
    assert result['due_wait_with_running_blocker_union_s'] == 1.
    assert result['due_wait_with_transit_blocker_union_s'] == 1.
    assert result['due_wait_with_known_blocker_union_s'] == 2.


@pytest.mark.parametrize('event', ['task', 'complete'])
def test_mixed_runs_with_duplicate_id_are_rejected(tmp_path, event):
    record = dict(event=event, task_id=1)
    for name in ('a', 'b'):
        (tmp_path/f'{name}.jsonl').write_text(json.dumps(record)+'\n')
    with pytest.raises(ValueError, match='isolated node/run'):
        analysis.analyze(tmp_path)


def test_missing_trace_directory_cannot_look_like_zero_wait(tmp_path):
    with pytest.raises(FileNotFoundError, match='No JSONL traces'):
        analysis.analyze(tmp_path/'missing')


def test_attempt_stage_totals_include_rejected_work_without_double_counting(tmp_path):
    record = dict(event='task', task_id=1, wall_s=12., cpu_s=11., components={},
        attempts=[dict(graph_s=3., sanity_s=1., result='class_sanity'),
                  dict(graph_s=2., sanity_s=1., filter_s=1., result='filter_rejected'),
                  dict(graph_s=1., sanity_s=1., filter_s=2., result='accepted')])
    (tmp_path/'worker.jsonl').write_text(json.dumps(record)+'\n')
    result = analysis.analyze(tmp_path)
    assert result['attempt_stage_wall_s'] == dict(graph_s=6., sanity_s=3., filter_s=3.)
    assert result['rejected_attempt_stage_wall_s'] == dict(graph_s=5., sanity_s=2., filter_s=1.)
    assert result['attempts_per_task']['max'] == 3
