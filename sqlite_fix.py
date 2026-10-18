"""chromadb を import する前に読み込むこと。

Streamlit Community Cloud などの Linux サーバーでは、OS 付属の sqlite3 が
chromadb の要求(3.35 以上)より古いことがあり、その場合 chromadb の import 時に
「Your system has an unsupported version of sqlite3」で落ちる。

requirements.txt で Linux のときだけ pysqlite3-binary(新しい sqlite3 を同梱)を
入れておき、ここで標準の sqlite3 と差し替える。
Windows ではこのパッケージが入らないので何もしない(Windows の Python は sqlite3 が新しい)。
"""

import sys

try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass
