import runpy
from pathlib import Path

ACT2_PATH = Path(__file__).parent / "act2.py"


def run_act2():
    runpy.run_path(str(ACT2_PATH), run_name="__main__")


def test_prints_hello_world(capsys):
    run_act2()
    assert capsys.readouterr().out == "Hello World\n"


def test_prints_exactly_one_line(capsys):
    run_act2()
    assert capsys.readouterr().out.splitlines() == ["Hello World"]


def test_writes_nothing_to_stderr(capsys):
    run_act2()
    assert capsys.readouterr().err == ""


def test_output_is_same_on_repeated_runs(capsys):
    run_act2()
    first = capsys.readouterr().out
    run_act2()
    second = capsys.readouterr().out
    assert first == second == "Hello World\n"
