# /// script
# requires-python = ">=3.10"
# dependencies = ["tomli; python_version < '3.11'"]
# ///
"""notes.toml に登録した LaTeX の文書をコンパイルし、公開用のファイルを _site/ にまとめる。

手元での実行:      uv run tools/build_site.py
GitHub Actions:    python3 tools/build_site.py
"""

import html
import shutil
import subprocess
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "_site"
REPO = "https://github.com/shiryu-nakano/tutorials"

docs = tomllib.loads((ROOT / "notes.toml").read_text(encoding="utf-8"))["doc"]

# ノート同士の参照を解決するため、全文書を2回ずつコンパイルする。
# 各ディレクトリの .latexmkrc（出力先などの設定）はそのまま使われる。
# -f を付けて、LaTeX のエラーがあっても PDF を作成できる限り処理を続ける。
# 2回目でもエラーが残った文書は、最後に一覧で表示する。
with_errors = []
for n in range(2):
    for d in docs:
        tex = ROOT / d["path"]
        result = subprocess.run(
            ["latexmk", "-lualatex", "-f", "-interaction=nonstopmode", tex.name],
            cwd=tex.parent,
        )
        if n == 1 and result.returncode != 0:
            with_errors.append(d["id"])

missing = [d["id"] for d in docs if not (ROOT / d["path"]).with_suffix(".pdf").exists()]
if missing:
    raise SystemExit(f"PDF を作成できなかった文書: {', '.join(missing)}")

shutil.rmtree(SITE, ignore_errors=True)
(SITE / "pdf").mkdir(parents=True)

rows = []
for d in docs:
    tex = ROOT / d["path"]
    shutil.copy(tex.with_suffix(".pdf"), SITE / "pdf" / f"{d['id']}.pdf")
    src = f"{REPO}/tree/main/{Path(d['path']).parent.as_posix()}"
    rows.append(
        "<tr>"
        f"<td>{html.escape(d['id'])}</td>"
        f'<td><a href="pdf/{html.escape(d["id"])}.pdf">{html.escape(d["title"])}</a></td>'
        f"<td>{html.escape(d['kind'])}</td>"
        f"<td>{html.escape(d['status'])}</td>"
        f'<td><a href="{src}">source</a></td>'
        "</tr>"
    )

(SITE / "index.html").write_text(
    f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Study Notes</title>
<style>
  body {{ font-family: sans-serif; max-width: 48rem; margin: auto; padding: 1rem; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ text-align: left; padding: 0.4rem 0.6rem; border-bottom: 1px solid #ddd; }}
</style>
</head>
<body>
<h1>Study Notes</h1>
<table>
<tr><th>ID</th><th>Title</th><th>Kind</th><th>Status</th><th>Source</th></tr>
{"".join(rows)}
</table>
</body>
</html>
""",
    encoding="utf-8",
)

print(f"{len(docs)} documents -> {SITE}")
if with_errors:
    print(f"warning: LaTeX のエラーが残っている文書: {', '.join(with_errors)}")
