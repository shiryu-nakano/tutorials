
# 編集方法
一覧ページに関わるファイルは二つです。

- `notes.toml`：どの文書を、どの見出しの下に並べるかを決めます。
- `tools/build_site.py`：`notes.toml` を読んで、一覧ページの HTML を書き出します。

見た目を変えやすくするため、CSS を `tools/style.css` という別のファイルに分けることを勧めます。以下は、その前提でまとめています。

## ① 見出しの整理

### `notes.toml`

各文書に `topic` を一行加えます。同じ `topic` の文書が、一つの見出しの下にまとまります。見出しは、`notes.toml` に初めて現れた順に並びます。

```toml
[[doc]]
id = "SLT1"
topic = "特異学習理論"
title = "Singular Learning Theory 1"
path = "singular_learning_theory/lecture_note/singular_learning_theory_1.tex"
kind = "note"
status = "draft"
```

`topic` を書かなかった文書は、最上位のディレクトリ名（`linear_algebra` など）を見出しとして使います。

### `tools/build_site.py`

`rows = []` の行から最後の `print(...)` の前までを、次の内容に置き換えます。

```python
groups = {}
for d in docs:
    topic = d.get("topic", Path(d["path"]).parts[0])
    groups.setdefault(topic, []).append(d)

sections = []
for topic, items in groups.items():
    rows = []
    for d in items:
        tex = ROOT / d["path"]
        shutil.copy(tex.with_suffix(".pdf"), SITE / "pdf" / f"{d['id']}.pdf")
        src = f"{REPO}/tree/main/{Path(d['path']).parent.as_posix()}"
        rows.append(
            "<tr>"
            f'<td><a href="pdf/{html.escape(d["id"])}.pdf">{html.escape(d["title"])}</a></td>'
            f"<td>{html.escape(d['kind'])}</td>"
            f"<td>{html.escape(d['status'])}</td>"
            f'<td><a href="{src}">source</a></td>'
            "</tr>"
        )
    sections.append(
        f"<section><h2>{html.escape(topic)}</h2>"
        "<table><tr><th>Title</th><th>Kind</th><th>Status</th><th>Source</th></tr>"
        f"{''.join(rows)}</table></section>"
    )

shutil.copy(ROOT / "tools" / "style.css", SITE / "style.css")

(SITE / "index.html").write_text(
    f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Study Notes</title>
<link rel="stylesheet" href="style.css">
</head>
<body>
<header><h1>Study Notes</h1></header>
<main>
{"".join(sections)}
</main>
</body>
</html>
""",
    encoding="utf-8",
)
```

見出しの下にさらに小見出しを付けたい場合は、`notes.toml` に `subtopic` のような項目を加え、同じ要領で二段階にまとめます。

## ② 見た目の変更

`tools/style.css` を新しく作り、見た目に関する指定はすべてここに書きます。HTML を作る処理とは分かれているので、このファイルだけを編集すれば見た目を変えられます。

```css
:root {
  --text: #222;
  --muted: #666;
  --accent: #2b6cb0;
  --line: #ddd;
  --bg: #fff;
}

body {
  font-family: system-ui, "Hiragino Sans", sans-serif;
  color: var(--text);
  background: var(--bg);
  max-width: 48rem;
  margin: 0 auto;
  padding: 1rem;
  line-height: 1.6;
}

h1 { font-size: 1.8rem; }
h2 { font-size: 1.2rem; margin-top: 2rem; border-bottom: 2px solid var(--accent); }

table { border-collapse: collapse; width: 100%; }
th, td { text-align: left; padding: 0.4rem 0.6rem; border-bottom: 1px solid var(--line); }
th { color: var(--muted); font-weight: normal; font-size: 0.9rem; }

a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
```

よく変える箇所は次のとおりです。

- **色**：先頭の `:root` の中の色を変えます。ホームページに合わせる場合は、ホームページの CSS から同じ色の値を写します。
- **フォント**：`body` の `font-family` を変えます。
- **ページの幅**：`body` の `max-width` を変えます。
- **見出しの飾り**：`h2` の `border-bottom` などを変えます。

## 変更を反映する手順

1. `notes.toml`、`tools/build_site.py`、`tools/style.css` を編集します。
2. `uv run tools/build_site.py` を実行し、`open _site/index.html` で手元の表示を確認します。
3. コミットして push すると、公開ページにも反映されます。















# めも
はい、uv ベースです。根拠は以下です。

pyproject.toml があり、[tool.poetry] や setup.py ではなく PEP 621 準拠の [project] テーブルで依存を定義している
ロックファイルが poetry.lock / Pipfile.lock ではなく uv.lock になっている（これは uv 固有）
.python-version に 3.10 が書かれており、uv が Python バージョン指定で参照する
典型的な使い方はこんな感じです。


# 依存インストール + venv 構築（uv.lock に従う）
```
uv sync
```
# main.py を venv の Python で実行
```
uv run python main.py
```
# 新しい依存を追加（pyproject.toml と uv.lock を自動更新）

```uv add scikit-learn```

# Jupyter を起動（notebook/ を使うなら）
uv run jupyter lab
uv sync するだけで .venv/ が uv.lock 通りに再現されるので、他マシンでの再現性はこれだけで足ります。


# ipynbをuvで動かすための手順

```
uv sync
```

```
uv add --dev jupyterlab ipykernel
```
ここで
- jupyterlabはノートブックUI本体
- ipykernelは作成した仮想環境.venvをJupyterのカーネルとして使えるようにするためのアダプタ→よくわからん


つぎに一応テスト用のipynbを作成するコマンドを．
```
mkdir -p notebook
uv run python -c "import nbformat as nbf, pathlib; nb=nbf.v4.new_notebook(); nb.cells=[nbf.v4.new_code_cell('import pymc as pm, arviz as az, numpy as np, japanize_matplotlib\nprint(pm.__version__)')]; pathlib.Path('notebook/intro.ipynb').write_text(nbf.writes(nb))"

```
その後vscodeでこれを開いて実行する．この時にカーネルとしてこのリポジトリで作成したカーネルを選択すること．

- [Visual Studio Codeにより起動したJupyter Notebookからuvで作成したカーネルを選択する手順 #Python - Qiita](https://qiita.com/daikikatsuragawa/items/bfa7314e3930856e6a83)

## 記録
実行したら，以下のエラーが出た
![alt text](image.png)

これは，

> 原因
> .venv の setuptools が 82.0.1（最新）
> setuptools 81 以降で pkg_resources モジュールが削除された
> しかし PyMC 5.7.2（2023年リリース）は古い API の pkg_resources を内部で使っている
> → import pymc が pkg_resources を探して失敗
> pyproject.toml:15 で setuptools をバージョン無指定にしているため、uv sync 時に最新版（= >
> pkg_resources 無し）が入ってしまったのが直接の引き金です。
ということなので，setuptoolsのバージョンを，最新にならないように指定した．
pytroject.tomlをで
setuptoops<81
と書いた．

---
