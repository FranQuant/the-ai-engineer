"""The rehearsal notebook is the experiment notebook with two lines changed
(DESIGN.md §9)."""

import ast
import json
import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
MAIN = HERE / "week03_fomc_surprise.ipynb"
REHEARSAL = HERE / "pilot" / "phase4_rehearsal.ipynb"


def _lines(path):
    nb = json.loads(path.read_text())
    return nb, [(i, line) for i, c in enumerate(nb["cells"])
                for line in c["source"]]


def test_rehearsal_differs_only_in_mode_and_ref():
    main, main_lines = _lines(MAIN)
    reh, reh_lines = _lines(REHEARSAL)
    assert main["metadata"] == reh["metadata"]
    assert [(c["cell_type"], c["id"]) for c in main["cells"]] == [
        (c["cell_type"], c["id"]) for c in reh["cells"]]
    assert len(main_lines) == len(reh_lines)
    diffs = [(a[1], b[1]) for a, b in zip(main_lines, reh_lines) if a != b]
    assert diffs == [('MODE = "real"\n', 'MODE = "rehearsal"\n'),
                     ('REF = "week03-v2-run1"\n',
                      'REF = "capstone/week03-v2"\n')]


def test_notebooks_are_saved_without_outputs():
    for path in (MAIN, REHEARSAL):
        nb = json.loads(path.read_text())
        code = [c for c in nb["cells"] if c["cell_type"] == "code"]
        assert all(c["outputs"] == [] and c["execution_count"] is None
                   for c in code), path.name


def _setup_function(name):
    """A function defined in the notebook's setup cell (cell-01)."""
    nb = json.loads(MAIN.read_text())
    src = "".join(next(c for c in nb["cells"] if c["id"] == "cell-01")
                  ["source"])
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)
              and n.name == name)
    ns = {"subprocess": subprocess, "Path": Path}
    exec(compile(ast.Module([fn], []), MAIN.name, "exec"), ns)
    return ns[name]


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_uncommitted_changes_reads_the_module_dir(tmp_path):
    check = _setup_function("uncommitted_changes")
    mod, other = tmp_path / "mod", tmp_path / "other"
    mod.mkdir()
    other.mkdir()
    assert check(mod) is None  # not a repository

    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=t",
                        "-c", "user.email=t@t", *args], check=True,
                       capture_output=True)

    git("init", "-q")
    (mod / "a.py").write_text("x = 1\n")
    (other / "b.py").write_text("y = 1\n")
    git("add", ".")
    git("commit", "-q", "-m", "init")
    assert check(mod) is False
    (other / "b.py").write_text("y = 2\n")  # outside the module dir
    assert check(mod) is False
    (mod / "new.py").write_text("")  # untracked counts
    assert check(mod) is True
    (mod / "new.py").unlink()
    (mod / "a.py").write_text("x = 2\n")  # modified counts
    assert check(mod) is True


FIX = "Use Runtime → Disconnect and delete runtime, then Run all again."


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t",
                           "-c", "user.email=t@t", "-c",
                           "advice.detachedHead=false", *args], check=True,
                          capture_output=True, text=True).stdout.strip()


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_real_mode_refusal(tmp_path):
    refusal = _setup_function("real_mode_refusal")
    origin, clone = tmp_path / "origin", tmp_path / "clone"
    origin.mkdir()
    _git(origin, "init", "-q")
    (origin / "a.py").write_text("x = 1\n")
    _git(origin, "add", ".")
    _git(origin, "commit", "-q", "-m", "one")
    _git(origin, "tag", "-a", "run1", "-m", "run1")  # annotated, as REF
    _git(origin, "commit", "-q", "--allow-empty", "-m", "two")
    # As the notebook clones: depth 1 at the tag, via a file:// URL so
    # --depth is honoured.
    _git(tmp_path, "clone", "-q", "--depth", "1", "--branch", "run1",
         origin.as_uri(), str(clone))

    assert refusal(clone, "clone", "run1") is None
    # Existing modules are refused even when they sit at REF.
    msg = refusal(clone, "local", "run1")
    assert msg and msg.endswith(FIX)
    # HEAD moved off REF.
    _git(clone, "commit", "-q", "--allow-empty", "-m", "three")
    msg = refusal(clone, "clone", "run1")
    assert msg and "run1" in msg and msg.endswith(FIX)
    # REF missing from the clone.
    msg = refusal(clone, "clone", "no-such-tag")
    assert msg and "no-such-tag is unresolved" in msg and msg.endswith(FIX)
    # Not a repository.
    msg = refusal(tmp_path, "clone", "run1")
    assert msg and msg.endswith(FIX)
