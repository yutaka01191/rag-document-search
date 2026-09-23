"""社内文書検索デモ(Streamlit)

このファイルは画面だけを担当する。RAGの処理は core.answer() に任せ、
ここにはロジックを書かない。Slackボットを作るときは、この app.py の代わりに
Slackのイベントハンドラから core.answer() を呼ぶだけでよい。

【起動】
    streamlit run app.py
"""

import time

import streamlit as st

import core

st.set_page_config(page_title="社内文書検索デモ", page_icon="📄", layout="centered")

SAMPLE_QUESTIONS = [
    "有給休暇は何日前までに申請しますか",
    "NCフライス盤2号機の異音は解決しましたか",
    "A社への納品ルールを教えてください",
]


@st.cache_resource(show_spinner="インデックスを読み込んでいます…")
def boot():
    """Chroma と LLM クライアントを1回だけ用意する。

    @st.cache_resource が無いと、画面を操作するたびに Chroma を開き直し、
    LLMクライアントを作り直すことになる。
    検索方式やkは毎回変えられるよう、ここではキャッシュしない。
    """
    core.warmup()
    return core.index_size()


# --- ヘッダ ------------------------------------------------------------------
st.title("社内文書検索デモ")
st.caption(
    "社内文書に対して質問すると、該当箇所を探して回答します。"
    "回答の根拠にした文書は下に表示されます。"
)

# --- 起動チェック ------------------------------------------------------------
try:
    chunk_count = boot()
except core.IndexNotFoundError as e:
    st.error(str(e))
    st.info("ターミナルで  python ingest.py  を実行してから、この画面を再読み込みしてください。")
    st.stop()
except Exception as e:
    st.error(f"起動に失敗しました: {e}")
    st.stop()

# --- 検索の設定(サイドバー) --------------------------------------------------
with st.sidebar:
    st.header("検索の設定")
    st.caption(f"索引済みチャンク数: {chunk_count}")

    search_type = st.radio(
        "検索方式",
        options=["mmr", "similarity"],
        index=0,
        help=(
            "similarity は質問に似ている順にk件取る。同じ話題が濃い1つの文書から "
            "似たチャンクばかり集まり、他の文書にある答えを取り逃がすことがある。\n\n"
            "mmr は広めに候補を取ったうえで、互いに重複しないものを選ぶ。"
            "出典が複数の文書に散るので、文書をまたぐ質問に強い。"
        ),
    )
    k = st.slider(
        "取得するチャンク数 k",
        min_value=1,
        max_value=10,
        value=core.DEFAULT_K,
        help="増やすほど answer に渡る情報が増えるが、関係ない情報も混ざりやすくなる。",
    )

    lambda_mult = core.DEFAULT_LAMBDA
    if search_type == "mmr":
        lambda_mult = st.slider(
            "関連度 ↔ 多様性 (lambda_mult)",
            min_value=0.0,
            max_value=1.0,
            value=core.DEFAULT_LAMBDA,
            step=0.1,
            help=(
                "1に近いほど質問との関連度を優先し、similarity の挙動に近づく。\n\n"
                "0に近いほど出典の多様性を優先するが、下げすぎると質問と関係のない "
                "文書まで混ざる。無関係な文書が出典に入るときは上げる。"
            ),
        )

    st.divider()
    st.caption(
        "「NCフライス盤2号機の異音は解決しましたか」を "
        "similarity / k=3 と mmr / k=5 で比べてみてください。"
        "出典がどの文書から集まるかが変わります。"
    )

# --- 質問欄 ------------------------------------------------------------------
if "q_input" not in st.session_state:
    st.session_state.q_input = ""

st.write("")
st.caption("よく使う質問")
cols = st.columns(len(SAMPLE_QUESTIONS))
for col, q in zip(cols, SAMPLE_QUESTIONS):
    if col.button(q, use_container_width=True, key=f"sample_{q}"):
        st.session_state.q_input = q

question = st.text_input(
    "質問",
    key="q_input",
    placeholder="例:経費精算の締切はいつですか",
)

run = st.button("回答する", type="primary")

# --- 回答 --------------------------------------------------------------------
if run:
    if not question.strip():
        st.warning("質問を入力してください。")
    else:
        started = time.perf_counter()
        try:
            with st.spinner("社内文書を検索しています…"):
                text, sources = core.answer(
                    question, k=k, search_type=search_type, lambda_mult=lambda_mult
                )
        except Exception as e:
            st.error(f"回答の生成に失敗しました: {e}")
        else:
            elapsed = time.perf_counter() - started

            st.markdown("### 回答")
            st.markdown(text)
            setting = f"検索方式 {search_type} / k={k}"
            if search_type == "mmr":
                setting += f" / lambda={lambda_mult:.1f}"
            st.caption(f"{elapsed:.1f} 秒 / {setting}")

            if sources:
                # 同じ文書の別々の箇所がヒットすることがある。中身は違うので
                # 検索結果からは削らず、表示だけ文書ごとにまとめる。
                # ここで1文書1件に絞ってしまうと、離れた2箇所に答えが分かれている
                # ケースで片方を捨てることになり、回答の材料が減る。
                grouped: dict[str, list[dict]] = {}
                for s in sources:
                    grouped.setdefault(s["title"], []).append(s)

                st.markdown("### 根拠にした箇所")
                st.caption(f"{len(grouped)} つの文書から {len(sources)} 箇所")

                for i, (title, items) in enumerate(grouped.items(), start=1):
                    ftype = items[0].get("filetype", "")
                    # 「[pdf](2箇所)」の形にすると Markdown のリンク記法として
                    # 解釈されてしまうため、角括弧と丸括弧を隣接させない。
                    label = f"{i}. {title}"
                    if ftype:
                        label += f"　{ftype}"
                    if len(items) > 1:
                        label += f"　{len(items)}箇所"
                    with st.expander(label):
                        for s in items:
                            # locator は PDF のページ番号、Excel のシート名。
                            # 「どこに書いてあったか」まで出せると商談での説得力が変わる。
                            if s.get("locator"):
                                st.caption(f"— {s['locator']}")
                            elif len(items) > 1:
                                st.caption("— 別の箇所")
                            st.write(s["snippet"])
                        if items[0]["filename"]:
                            st.caption(f"ファイル: {items[0]['filename']}")
            else:
                st.info("該当する記載が社内文書に見つかりませんでした。")

st.divider()
st.caption("文書を追加・変更したときは  python ingest.py  を再実行してください。")
