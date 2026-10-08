"""Regenerate the student files (model_zoo/m1-m6) from solutions/. For CV leads.

    python3 tools/make_exercises.py

In each solution file, every block between '# ANSWER-START' and '# ANSWER-END' becomes a TODO that
raises NotImplementedError. The comments above each block (what to write and why) stay. Lines
that contain 'solutions-only' are dropped. So: edit the solution, then regenerate. Never edit the
generated files by hand, and don't run this in a checkout where someone is doing the exercises.
"""
from pathlib import Path

ZOO = Path(__file__).resolve().parents[1]
FILES = ["m1_io_cards.py", "m2_yolov5.py", "m3_yolox.py", "m4_ultralytics.py", "m5_end2end.py", "m6_same_network.py"]


def to_exercise(text):
    out, in_answer = [], False
    for line in text.splitlines():
        stripped = line.strip()
        if "solutions-only" in line:
            continue
        if stripped == "# ANSWER-START":
            indent = line[:len(line) - len(line.lstrip())]
            out += [f"{indent}# TODO", f"{indent}raise NotImplementedError"]
            in_answer = True
        elif stripped == "# ANSWER-END":
            in_answer = False
        elif not in_answer:
            out.append(line)
    if in_answer:
        raise ValueError("an ANSWER-START has no ANSWER-END")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    for name in FILES:
        text = (ZOO / "solutions" / name).read_text(encoding="utf-8")
        (ZOO / name).write_text(to_exercise(text), encoding="utf-8")
        print(f"wrote {name}  ({text.count('# ANSWER-START')} TODOs)")
