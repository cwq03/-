"""Web prototype for sentence semantic similarity."""

from __future__ import annotations

import html
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

from nlp_similarity import SentenceSimilarityEngine
from nlp_similarity.ml_model import DEFAULT_MODEL_PATH, LogisticSimilarityModel


HOST = "127.0.0.1"
PORT = 8000
ENGINE = SentenceSimilarityEngine()
ML_MODEL = LogisticSimilarityModel.load(DEFAULT_MODEL_PATH) if DEFAULT_MODEL_PATH.exists() else None


PAGE_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>句子语义相似度计算系统</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #1f2937;
      --muted: #64748b;
      --line: #d7dee8;
      --panel: #ffffff;
      --bg: #f4f7fb;
      --accent: #2563eb;
      --accent-dark: #1d4ed8;
      --ok: #047857;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }}
    header {{
      padding: 28px 24px 18px;
      border-bottom: 1px solid var(--line);
      background: #ffffff;
    }}
    main {{
      max-width: 1080px;
      margin: 0 auto;
      padding: 24px;
    }}
    h1 {{
      max-width: 1080px;
      margin: 0 auto 8px;
      font-size: 28px;
      letter-spacing: 0;
    }}
    .subtitle {{
      max-width: 1080px;
      margin: 0 auto;
      color: var(--muted);
      line-height: 1.6;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }}
    form, .result, .explain {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
    }}
    form {{
      margin-bottom: 16px;
    }}
    label {{
      display: block;
      margin-bottom: 8px;
      font-weight: 700;
    }}
    textarea {{
      width: 100%;
      min-height: 124px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 12px;
      font: inherit;
      line-height: 1.5;
      background: #fbfdff;
    }}
    button {{
      margin-top: 16px;
      border: 0;
      border-radius: 6px;
      padding: 11px 18px;
      color: #fff;
      background: var(--accent);
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }}
    button:hover {{ background: var(--accent-dark); }}
    .score {{
      display: flex;
      align-items: baseline;
      gap: 14px;
      margin-bottom: 14px;
    }}
    .score strong {{
      font-size: 42px;
      color: var(--ok);
    }}
    .badge {{
      border: 1px solid #bbf7d0;
      background: #ecfdf5;
      color: #047857;
      border-radius: 999px;
      padding: 5px 10px;
      font-weight: 700;
    }}
    .bar {{
      height: 10px;
      background: #e5eaf1;
      border-radius: 999px;
      overflow: hidden;
      margin: 12px 0 18px;
    }}
    .bar span {{
      display: block;
      height: 100%;
      width: {score_percent}%;
      background: var(--accent);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 8px;
    }}
    th, td {{
      text-align: left;
      border-bottom: 1px solid var(--line);
      padding: 10px 8px;
    }}
    th {{ color: var(--muted); }}
    .tokens {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 8px;
    }}
    .token {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 5px 9px;
      background: #f8fafc;
      font-size: 14px;
    }}
    .empty {{
      color: var(--muted);
      line-height: 1.7;
    }}
    @media (max-width: 760px) {{
      .grid {{ grid-template-columns: 1fr; }}
      main {{ padding: 16px; }}
      header {{ padding-left: 16px; padding-right: 16px; }}
      .score strong {{ font-size: 34px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>句子语义相似度计算系统</h1>
    <p class="subtitle">系统提供两种核心模型：规则多特征融合模型与监督学习融合模型。输入两个句子后，系统会对比 TF-IDF 基线、同义词规则、规则融合分数和监督模型概率，并输出可解释的语义相似度结果。</p>
  </header>
  <main>
    <form method="post">
      <div class="grid">
        <div>
          <label for="sentence_a">句子 A</label>
          <textarea id="sentence_a" name="sentence_a">{sentence_a}</textarea>
        </div>
        <div>
          <label for="sentence_b">句子 B</label>
          <textarea id="sentence_b" name="sentence_b">{sentence_b}</textarea>
        </div>
      </div>
      <button type="submit">计算相似度</button>
    </form>
    {result_html}
  </main>
</body>
</html>
"""


class SimilarityHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self._send_page("", "")

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        fields = parse_qs(body)
        sentence_a = fields.get("sentence_a", [""])[0]
        sentence_b = fields.get("sentence_b", [""])[0]
        self._send_page(sentence_a, sentence_b)

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_page(self, sentence_a: str, sentence_b: str) -> None:
        result_html = render_result(sentence_a, sentence_b) if sentence_a or sentence_b else render_empty()
        score_percent = "0"
        if sentence_a or sentence_b:
            score_percent = str(round(ENGINE.compare(sentence_a, sentence_b).score * 100, 1))
        page = PAGE_TEMPLATE.format(
            sentence_a=html.escape(sentence_a),
            sentence_b=html.escape(sentence_b),
            result_html=result_html,
            score_percent=score_percent,
        )
        encoded = page.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def render_empty() -> str:
    return """
    <section class="result">
      <p class="empty">可以试试示例：A「我喜欢自然语言处理」，B「我对NLP很感兴趣」。</p>
    </section>
    """


def render_result(sentence_a: str, sentence_b: str) -> str:
    result = ENGINE.compare(sentence_a, sentence_b)
    methods = [
        ("方法一：TF-IDF 词面基线", "基础词面相似", result.lexical_baseline),
        ("方法二：同义词归一化 + Jaccard", "简单语义规则", result.semantic_rule_score),
        ("方法三：多特征融合相似度", "最终系统结果", result.score),
    ]
    if ML_MODEL is not None:
        prediction = ML_MODEL.predict(result)
        methods.append(
            (
                "方法四：监督学习融合模型",
                f"Logistic Regression，预测：{'相似' if prediction.label else '不相似'}",
                prediction.probability,
            )
        )
    features = [
        ("词级 TF-IDF 余弦相似度", result.word_tfidf),
        ("字符 n-gram TF-IDF 余弦相似度", result.char_ngram_tfidf),
        ("同义词归一化 Jaccard", result.normalized_jaccard),
        ("编辑相似度", result.edit_similarity),
        ("否定极性惩罚系数", result.negation_penalty),
        ("反义极性惩罚系数", result.opposite_penalty),
    ]
    method_rows = "\n".join(
        f"<tr><td>{html.escape(name)}</td><td>{html.escape(note)}</td><td>{value:.4f}</td></tr>"
        for name, note, value in methods
    )
    feature_rows = "\n".join(
        f"<tr><td>{html.escape(name)}</td><td>{value:.4f}</td></tr>" for name, value in features
    )
    tokens_a = render_tokens(result.normalized_tokens_a)
    tokens_b = render_tokens(result.normalized_tokens_b)
    raw_json = html.escape(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return f"""
    <section class="result">
      <div class="score">
        <strong>{result.score:.4f}</strong>
        <span class="badge">{html.escape(result.label)}</span>
      </div>
      <div class="bar"><span></span></div>
      <table>
        <thead><tr><th>方法</th><th>含义</th><th>分数</th></tr></thead>
        <tbody>{method_rows}</tbody>
      </table>
      <table>
        <thead><tr><th>特征</th><th>分数</th></tr></thead>
        <tbody>{feature_rows}</tbody>
      </table>
      <p class="empty">否定极性不一致：{"是" if result.negation_mismatch else "否"}；反义极性冲突：{"是" if result.opposite_mismatch else "否"}；惩罚前融合分数：{result.fusion_before_penalty:.4f}</p>
    </section>
    <section class="explain" style="margin-top:16px">
      <div class="grid">
        <div>
          <h2>句子 A 归一化词元</h2>
          {tokens_a}
        </div>
        <div>
          <h2>句子 B 归一化词元</h2>
          {tokens_b}
        </div>
      </div>
      <details style="margin-top:16px">
        <summary>查看完整 JSON 结果</summary>
        <pre>{raw_json}</pre>
      </details>
    </section>
    """


def render_tokens(tokens: list[str]) -> str:
    if not tokens:
        return '<p class="empty">没有可用词元</p>'
    items = "".join(f'<span class="token">{html.escape(token)}</span>' for token in tokens)
    return f'<div class="tokens">{items}</div>'


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), SimilarityHandler)
    print(f"句子语义相似度计算系统已启动：http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止")


if __name__ == "__main__":
    main()
