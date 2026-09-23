"""RAGの中核。外に公開するのは answer() ひとつだけ。

【この境界を作る理由】
app.py(Streamlit)も、将来つくる Slack ボットも、Microsoft Teams 版も、
呼ぶのは answer() だけにする。そうしておけば移植で書き直すのは
「入口と出口」だけで済み、この中核は一切触らずに使い回せる。

    answer("有給は何日前までに申請?") -> ("3営業日前までに…", [出典1, 出典2, 出典3])
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

BASE_DIR = Path(__file__).parent
PERSIST_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "shanai_docs"

# --- 検索の既定値 -------------------------------------------------------------
# search_type:
#   "similarity" … 質問に似ている順にk件。素直だが、同じ話題が濃い1文書から
#                  似たチャンクばかり集まり、他の文書に書かれた答えを取り逃がす。
#   "mmr"        … まず fetch_k 件を広めに取り、その中から「質問に近い」かつ
#                  「互いに重複しない」ものをk件選ぶ。出典が複数文書に散るので、
#                  複数の文書をまたぐ質問に強い。
DEFAULT_SEARCH_TYPE = "mmr"
DEFAULT_K = 5
DEFAULT_FETCH_K = 20
DEFAULT_LAMBDA = 0.5  # 0に近いほど多様性重視、1に近いほど関連度重視

SNIPPET_CHARS = 200

PROMPT = ChatPromptTemplate.from_template(
    "あなたは社内文書の内容だけを使って質問に答えるアシスタントです。\n"
    "以下のルールを守ってください。\n"
    "- コンテキストに書かれていないことは答えず、「社内文書からは分かりません」と答える\n"
    "- 推測で補わない\n"
    "- 複数の文書にまたがる情報は、時系列や経緯が分かるようにつないで答える\n"
    "- 日本語で、結論から簡潔に答える\n"
    "- 数値や期限は文書の表記をそのまま使う\n\n"
    "コンテキスト:\n{context}\n\n"
    "質問: {question}"
)


class IndexNotFoundError(RuntimeError):
    """Chroma のインデックスがまだ作られていない。"""


# --- 遅延初期化 --------------------------------------------------------------
# 重いのは「Chromaを開く」「LLMクライアントを作る」の2つだけなので、
# そこをモジュール変数に保持する。Retriever は毎回作っても安いため、
# 検索方式やkを画面から切り替えられるように answer() の中で組み立てる。
_vectorstore = None
_chain = None


def _load_env_or_raise() -> None:
    load_dotenv(BASE_DIR / ".env")
    missing = [k for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY") if not os.getenv(k)]
    if missing:
        raise RuntimeError(f".env に次のキーが設定されていません: {', '.join(missing)}")


def warmup() -> None:
    """VectorStore と Chain を組み立てる。最初の質問の前に一度呼ぶ。"""
    global _vectorstore, _chain

    if _vectorstore is not None and _chain is not None:
        return

    _load_env_or_raise()

    if not PERSIST_DIR.exists():
        raise IndexNotFoundError(
            f"インデックスが見つかりません: {PERSIST_DIR}\n"
            "先に  python ingest.py  を実行してください。"
        )

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    _vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(PERSIST_DIR),
    )

    if _vectorstore._collection.count() == 0:
        raise IndexNotFoundError(
            "インデックスが空です。先に  python ingest.py  を実行してください。"
        )

    llm = ChatAnthropic(model="claude-sonnet-4-5", temperature=0)
    # ここに retriever を含めない点が重要。
    # retriever | prompt | llm | StrOutputParser() と一本に繋ぐと
    # 返ってくるのは回答の文字列だけで、検索された Document が手元に残らず、
    # 出典を表示できない。検索は answer() の中で別に呼び、
    # Chain は「コンテキストと質問 → 回答文」だけを担当させる。
    _chain = PROMPT | llm | StrOutputParser()


def index_size() -> int:
    """索引に入っているチャンク数。動作確認用。"""
    warmup()
    return _vectorstore._collection.count()


def _build_retriever(search_type: str, k: int, lambda_mult: float = DEFAULT_LAMBDA):
    if search_type == "mmr":
        return _vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": DEFAULT_FETCH_K, "lambda_mult": lambda_mult},
        )
    return _vectorstore.as_retriever(search_kwargs={"k": k})


def _format_docs(docs: list[Document]) -> str:
    """LLMに渡すコンテキスト。どの文書からの抜粋かを明示しておく。"""
    blocks = []
    for doc in docs:
        title = doc.metadata.get("title", "不明な文書")
        blocks.append(f"【{title}】\n{doc.page_content}")
    return "\n\n".join(blocks)


def _to_source(doc: Document) -> dict:
    snippet = doc.page_content.strip().replace("\n", " ")
    if len(snippet) > SNIPPET_CHARS:
        snippet = snippet[:SNIPPET_CHARS] + "…"
    return {
        "title": doc.metadata.get("title", "不明な文書"),
        "filename": doc.metadata.get("filename", ""),
        "filetype": doc.metadata.get("filetype", ""),
        # PDFならページ番号、Excelならシート名。商談で「○○の3ページ目です」と言える。
        "locator": doc.metadata.get("locator", ""),
        "snippet": snippet,
    }


def search_only(
    question: str,
    k: int = DEFAULT_K,
    search_type: str = DEFAULT_SEARCH_TYPE,
    lambda_mult: float = DEFAULT_LAMBDA,
) -> list[dict]:
    """回答を作らず、検索結果だけを返す。検索の効き具合を確かめるとき用。"""
    warmup()
    docs = _build_retriever(search_type, k, lambda_mult).invoke(question)
    return [_to_source(d) for d in docs]


def answer(
    question: str,
    k: int = DEFAULT_K,
    search_type: str = DEFAULT_SEARCH_TYPE,
    lambda_mult: float = DEFAULT_LAMBDA,
) -> tuple[str, list[dict]]:
    """質問に回答し、根拠にした文書の抜粋を一緒に返す。

    Returns:
        (回答文, 出典のリスト)
        出典は {"title", "filename", "snippet"} の辞書。
    """
    warmup()

    # 1. まず検索する。Document をここで受け取るので出典が手元に残る。
    docs = _build_retriever(search_type, k, lambda_mult).invoke(question)

    if not docs:
        return "社内文書からは分かりません。", []

    # 2. コンテキストを組んで回答を生成する。
    text = _chain.invoke({"context": _format_docs(docs), "question": question})

    # 3. 画面表示用に出典を整える。
    return text, [_to_source(d) for d in docs]


if __name__ == "__main__":
    # 画面なしで動作確認したいとき:
    #   python core.py "質問文"
    #   python core.py "質問文" similarity 3     ← 検索方式とkを指定して比較する
    q = sys.argv[1] if len(sys.argv) > 1 else "有給休暇は何日前までに申請しますか"
    st_ = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_SEARCH_TYPE
    kk = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_K

    print(f"索引チャンク数: {index_size()}")
    print(f"検索方式: {st_} / k={kk}\n")

    text, sources = answer(q, k=kk, search_type=st_)
    print(f"Q: {q}\n")
    print(f"A: {text}\n")
    print("--- 出典 ---")
    for i, s in enumerate(sources, start=1):
        where = f" / {s['locator']}" if s["locator"] else ""
        print(f"{i}. {s['title']}{where}  ({s['filename']})")
        print(f"   {s['snippet']}\n")
