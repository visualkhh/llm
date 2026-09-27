"""laya-serve 실행 포인트 — 다른 디렉토리(chat/, 3d/)와 똑같이
`uv run python3 laya/run.py`로 띄울 수 있게 하는 얇은 래퍼.

laya 패키지가 설치할 때 `.venv/bin/laya-serve`라는 CLI를 자체적으로 만들어
주기 때문에 원래는 이 파일 없이 `uv run laya-serve`만 쳐도 되지만(README
참고), 이 저장소의 다른 하위 프로젝트들처럼 '이 디렉토리 안에 실행할
.py가 있는' 형태로 맞추기 위한 용도. 실제 서버 로직은 전부 laya.serve에
있고, 여기서는 그걸 그대로 부르기만 한다.

실행: uv run python3 laya/run.py
      LAYA_PRELOAD=0 uv run python3 laya/run.py   (빠른 테스트용, 지연 로드)
"""
from laya.serve import main

if __name__ == "__main__":
    main()
