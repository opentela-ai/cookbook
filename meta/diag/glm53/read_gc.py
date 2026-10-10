#!/usr/bin/env python3
"""Read a gen_correctness JSON and print verdict + per-request detail."""
import json
import sys


def main():
    for path in sys.argv[1:]:
        try:
            m = json.load(open(path))
        except Exception as exc:  # noqa: BLE001
            print(f"=== {path}: ERROR {exc!r}")
            continue
        print(f"=== {path}")
        print(f"  verdict={m.get('verdict')} pass={m.get('pass')} "
              f"crisp={m.get('crisp_pass')}/{m.get('crisp_total')} "
              f"elapsed={m.get('elapsed_s')}s")
        print()
        for r in m.get("results", []):
            tag = "OK " if r.get("ok") else "XX "
            crisp = r.get("crisp")
            label = r.get("label")
            prompt = r.get("prompt")
            text = r.get("text")
            err = r.get("error", "")
            print(f"  [{tag}] crisp={crisp!r:5} label={label!r}")
            print(f"        Q: {prompt!r}")
            print(f"        A: {text!r}  err={err!r}")
        print()


if __name__ == "__main__":
    main()
