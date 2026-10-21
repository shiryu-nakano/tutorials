# DeepLearning_tutorial

体系的に勉強したことのまとめ．

[MCMCの勉強](./bayes_mcmc/README.md)





- Pandas Tutorial
    - https://drive.google.com/file/d/13qSySGGnkBM9IsnNGTgp1CSg5jplIDUY/view?usp=sharing
    - 


# 勉強することリスト
- https://shiryu-nakano-github-io.pages.dev/posts/study/overview/
- https://docs.google.com/spreadsheets/d/1bw45aQ0Vq8PmYOUlPyN4suDOcVb9p1iYJ9yodbQxANo/edit?usp=sharing
    - 随時追加すること





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
