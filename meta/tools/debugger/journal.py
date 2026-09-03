#!/usr/bin/env python3
"""journal — investigation journal for a correctness bug (all phases).

Creates and appends to the structured investigation doc the GLM-5.3-Flash
MI300A campaign converged on (see CORRECTNESS_BUG_INVESTIGATION.md): a precise
symptom characterization, a RULED OUT list where every closure carries the
evidence that closed it, a narrowed bug statement, ranked suspects, next steps,
and operational notes. The discipline matters: record the verdict WHEN you get
it, with the test that produced it — the ruled-out list is what keeps a
multi-day bisect sane and turns the doc into the eventual commit message.

Usage:
  python3 journal.py new INV.md --title "GLM-X garbage output on SITE" \
      --symptom "real tokens, weakly-peaked logprobs, context-present-but-wrong"
  python3 journal.py ruled-out INV.md "weight loading" \
      --evidence "no missing keys / shape mismatch / OOM in load section"
  python3 journal.py suspect   INV.md "MHC pre-norm tilelang splitk" --rank high \
      --evidence "never validated after hidden_block 256->128 LDS reduction"
  python3 journal.py verdict   INV.md "forward-pass numerical error, novel kernel" \
      --evidence "load/FP8/template/bigram ruled out; lstats first bad layer = 3"
  python3 journal.py next      INV.md "single-expert FP8 GEMM primitive test"
  python3 journal.py note      INV.md "EDF env clobbers sbatch exports; pass inline"

All subcommands except `new` take the journal path FIRST. Entries are dated
bullets inserted under the right section; sections are created if missing.
Stdlib-only; exit 0 on success, 2 on usage errors.
"""
from __future__ import annotations

import argparse
import datetime
import sys

TEMPLATE = """# {title} — correctness bug investigation

Started {date}. Tooling: `meta/tools/debugger/` (method in its README).

## Symptom (characterized, not paraphrased)

{symptom}

Signature notes (probe.py): logprob peak, determinism, context-sensitivity,
concurrency — see the failure-signature table in meta/tools/debugger/README.md.

## What is RULED OUT (with evidence)

(each entry: the class, the EXACT test that closed it, and the observed result)

## What the bug is (narrowed)

## Suspects (ranked)

## Next steps

## Operational notes (site quirks, env gotchas, budgets)
"""

SECTIONS = {
    "ruled-out": "## What is RULED OUT (with evidence)",
    "verdict": "## What the bug is (narrowed)",
    "suspect": "## Suspects (ranked)",
    "next": "## Next steps",
    "note": "## Operational notes (site quirks, env gotchas, budgets)",
}


def _read(path):
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        print(f"error: journal {path} does not exist (run `journal.py new` first)", file=sys.stderr)
        sys.exit(2)


def _write(path, text):
    with open(path, "w") as f:
        f.write(text)


def cmd_new(args):
    if not args.force and _exists(args.path):
        print(f"error: {args.path} exists; refusing to overwrite (use --force)", file=sys.stderr)
        sys.exit(2)
    title = args.title or args.path.rsplit("/", 1)[-1].replace(".md", "")
    _write(args.path, TEMPLATE.format(
        title=title, date=datetime.date.today().isoformat(),
        symptom=args.symptom or "(fill in: what exactly is wrong? real tokens? peak of logprobs? deterministic?)"))
    print(f"journal created: {args.path}")
    print("rule of thumb: one `ruled-out` entry per closed class, WITH the test that closed it.")


def _exists(path):
    try:
        with open(path):
            return True
    except FileNotFoundError:
        return False


def _insert_under(text, heading, bullet):
    """Insert bullet at the END of `heading`'s section (before next '## ')."""
    lines = text.splitlines(keepends=True)
    start = None
    for i, ln in enumerate(lines):
        if ln.rstrip() == heading:
            start = i + 1
            break
    if start is None:  # section missing: append it
        return text.rstrip() + f"\n\n{heading}\n\n{bullet}\n"
    end = len(lines)
    for j in range(start, len(lines)):
        if lines[j].startswith("## "):
            end = j
            break
    # skip a trailing blank block to keep sections tight
    while end - 1 > start and lines[end - 1].strip() == "":
        end -= 1
    lines.insert(end, bullet if bullet.endswith("\n") else bullet + "\n")
    return "".join(lines)


def cmd_append(args):
    text = _read(args.path)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    if args.kind == "ruled-out":
        bullet = f"- **{args.claim}** — RULED OUT ({ts}). {args.evidence or '(no evidence recorded — do not skip this)'}"
    elif args.kind == "suspect":
        rank = f" [{args.rank.upper()}]" if args.rank else ""
        bullet = f"- {rank} **{args.claim}** ({ts}). {args.evidence or ''}".rstrip()
    elif args.kind == "verdict":
        bullet = f"- **{args.claim}** ({ts}). {args.evidence or ''}".rstrip()
    elif args.kind == "next":
        bullet = f"- [ ] {args.claim} ({ts})"
    else:  # note
        bullet = f"- {args.claim} ({ts})"
    _write(args.path, _insert_under(text, SECTIONS[args.kind], bullet))
    print(f"journal updated: {args.kind} -> {SECTIONS[args.kind]}: {args.claim}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="kind", required=True)

    pn = sub.add_parser("new", help="create a new investigation journal")
    pn.add_argument("path")
    pn.add_argument("--title")
    pn.add_argument("--symptom", help="the characterized failure signature")
    pn.add_argument("--force", action="store_true")
    pn.set_defaults(fn=cmd_new)

    for kind, helptext in (
        ("ruled-out", "record a ruled-out class (claim + the test that closed it)"),
        ("suspect", "record a ranked suspect"),
        ("verdict", "record a narrowed-bug finding"),
        ("next", "record a next step"),
        ("note", "record an operational note"),
    ):
        p = sub.add_parser(kind, help=helptext)
        p.add_argument("path")
        p.add_argument("claim")
        p.add_argument("--evidence", help="the evidence / test result backing this entry")
        p.add_argument("--rank", choices=["high", "medium", "low"], help="suspect rank only")
        p.set_defaults(fn=cmd_append)

    args = ap.parse_args()
    if args.kind != "new" and args.rank and args.kind != "suspect":
        print("note: --rank only applies to suspects", file=sys.stderr)
    args.fn(args)


if __name__ == "__main__":
    main()
