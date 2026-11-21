"""Run the eval suite against one or more backends and write report.html.

    python report/generate_report.py guarded naive
    python report/generate_report.py groq
"""
import html
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals.runner import load_cases, run_case  # noqa: E402

OUTPUT = Path(__file__).parent / "report.html"

STYLE = """
body { font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 960px; color: #1c1c1c; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; }
th, td { border: 1px solid #d0d0d0; padding: 6px 10px; text-align: left; vertical-align: top; }
th { background: #f2f2f2; }
.pass { color: #1a7f37; font-weight: 600; }
.fail { color: #b42318; font-weight: 600; }
code { background: #f5f5f5; padding: 1px 4px; }
"""


def evaluate(backend, cases):
    results = []
    for case in cases:
        log, failures = run_case(case, backend)
        results.append({"case": case, "log": log, "failures": failures})
    return results


def summary_rows(results):
    by_category = defaultdict(lambda: [0, 0])
    for result in results:
        counts = by_category[result["case"]["category"]]
        counts[1] += 1
        counts[0] += not result["failures"]
    return by_category


def render(all_results):
    parts = [f"<html><head><meta charset='utf-8'><title>Agent eval report</title><style>{STYLE}</style></head><body>"]
    parts.append("<h1>Dealership agent evaluation report</h1>")
    for backend, results in all_results.items():
        passed = sum(not r["failures"] for r in results)
        parts.append(f"<h2>Backend: {html.escape(backend)} &mdash; {passed}/{len(results)} passed</h2>")
        parts.append("<table><tr><th>Category</th><th>Passed</th><th>Total</th></tr>")
        for category, (ok, total) in sorted(summary_rows(results).items()):
            css = "pass" if ok == total else "fail"
            parts.append(f"<tr><td>{html.escape(category)}</td><td class='{css}'>{ok}</td><td>{total}</td></tr>")
        parts.append("</table>")
        failed = [r for r in results if r["failures"]]
        if failed:
            parts.append("<table><tr><th>Failed case</th><th>Message</th><th>Reply</th><th>Why it failed</th></tr>")
            for result in failed:
                reasons = "<br>".join(html.escape(f) for f in result["failures"])
                parts.append(
                    "<tr>"
                    f"<td><code>{html.escape(result['case']['id'])}</code></td>"
                    f"<td>{html.escape(result['log'].message)}</td>"
                    f"<td>{html.escape(result['log'].reply)}</td>"
                    f"<td>{reasons}</td>"
                    "</tr>"
                )
            parts.append("</table>")
    parts.append("</body></html>")
    return "\n".join(parts)


def main():
    backends = sys.argv[1:] or ["guarded", "naive"]
    cases = load_cases()
    all_results = {backend: evaluate(backend, cases) for backend in backends}
    OUTPUT.write_text(render(all_results))
    for backend, results in all_results.items():
        passed = sum(not r["failures"] for r in results)
        print(f"{backend}: {passed}/{len(results)} passed")
        for category, (ok, total) in sorted(summary_rows(results).items()):
            print(f"  {category:<18} {ok}/{total}")
    print(f"Report written to {OUTPUT}")


if __name__ == "__main__":
    main()
