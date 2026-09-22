"""A failed cutover must hand back its CAUSE, not only the stable outer message (2026-09-22)."""
from __future__ import annotations

import json
from types import SimpleNamespace

from gpu_worker import manage_worker_release as m
from gpu_worker.worker_release import WorkerReleaseError


def test_cli_prints_the_cause_of_a_failed_cutover(monkeypatch, capsys):
    def handler(_args):
        try:
            raise WorkerReleaseError("worker port 9001 has no listener\n--- journalctl -u filmforge-worker-edge-gpu1.service ---\nTraceback: boom")
        except WorkerReleaseError as inner:
            raise WorkerReleaseError("secure cutover failed; the prior safe state was restored") from inner

    monkeypatch.setattr(m, "parse_args", lambda argv: SimpleNamespace(handler=handler))
    assert m.main([]) == 2
    out = json.loads(capsys.readouterr().out)
    assert out["error"] == "secure cutover failed; the prior safe state was restored"
    assert "worker port 9001" in out["cause"] and "Traceback: boom" in out["cause"]


def test_cli_omits_cause_when_there_is_none(monkeypatch, capsys):
    def handler(_args):
        raise WorkerReleaseError("invalid secure-profile release id")

    monkeypatch.setattr(m, "parse_args", lambda argv: SimpleNamespace(handler=handler))
    assert m.main([]) == 2
    assert "cause" not in json.loads(capsys.readouterr().out)


def test_one_click_surfaces_the_remote_cause_past_the_stdout_tail():
    from gpu_worker import secure_one_click as s

    journal = "worker port 9001 has no listener\n" + "x" * 500 + "\nTraceback: boom"
    stdout = "noise\n" + json.dumps({"ok": False, "error": "secure cutover failed; the prior safe state was restored",
                                     "cause": journal}, sort_keys=True)
    out = s._remote_failure_cause(stdout)
    assert out.startswith("\n[remote cause]\n") and "Traceback: boom" in out and "9001" in out
    assert s._remote_failure_cause('{"ok": false, "error": "x"}') == ""
    assert s._remote_failure_cause("not json") == ""
