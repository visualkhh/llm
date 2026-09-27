# laya — PyPI `laya[serve]` 그대로 사용

`laya`는 자유 질의응답이 아니라, **choice(분류)/score(등급)/noul(예/아니오 확률)**
같은 정해진 형태의 판단을 100개 이상 언어에서 빠르게 내려주는 결정 엔진이다
([PyPI](https://pypi.org/project/laya/), [GitHub](https://github.com/NandhaKishorM/laya)).
직접 만들지 않고 그 패키지를 그대로 설치해서 쓴다 — `uv add "laya[serve]"`로
이미 `pyproject.toml`에 들어가 있음.

## 실행

```
uv run python3 laya/run.py
```

(`chat/`, `3d/`처럼 이 저장소의 다른 하위 프로젝트와 동일하게 디렉토리 안에
실행 파일을 두는 형태. `run.py`는 laya 패키지가 자체 설치하는 CLI
`laya-serve`(`.venv/bin/laya-serve`)를 그대로 부르는 얇은 래퍼라, 원한다면
`uv run laya-serve`를 직접 써도 동일하게 동작한다.)

기본값: `http://0.0.0.0:8000`, 시작 시 체크포인트 3개(english/multilingual/
typed-decisions)를 전부 미리 로드(`LAYA_PRELOAD=1`)해서 첫 요청이 느리지 않게
한다 — 대신 뜨는 데 시간이 좀 걸리고 각 모델(3~4억 파라미터급)을 처음
받을 때 네트워크 다운로드가 필요하다. 빨리 테스트만 해보고 싶으면:

```
LAYA_PRELOAD=0 uv run laya-serve   # 요청 올 때 필요한 모델만 그때 로드
```

주요 환경변수: `LAYA_HOST`, `LAYA_PORT`(기본 8000), `LAYA_DEVICE`(cpu/cuda/mps),
`LAYA_MODELS`(콤마로 미리 로드할 것만 지정), `LAYA_API_KEY`(설정하면 Bearer 인증 요구).

## 호출 (실제로 확인된 엔드포인트)

⚠️ 설치된 버전(0.3.20)은 REST 경로가 `/predict`가 아니라 **`/v1/systemone`**
이다(TypeSafe Jev 프로토콜과 호환되게 만든 것 — `laya/serve.py` 소스 직접
확인함, 인터넷 문서가 버전에 따라 다르게 나와서 실제 설치된 코드 기준으로 씀).

```bash
curl -s http://localhost:8000/health

curl -s -X POST http://localhost:8000/v1/systemone \
  -H 'content-type: application/json' \
  -d '{
    "state": "3월에 요금이 두 번 청구됐어요. 오늘 중복 환불 부탁드려요.",
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "어느 부서가 처리해야 하나요?",
        "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages", "other": "everything else"}
      },
      "urgency": {
        "type": "score",
        "instructions": "얼마나 긴급한가요?",
        "criteria": ["not urgent", "soon", "critical"]
      }
    }
  }'
```

응답에는 각 질문의 답 + 확률분포 + 신뢰도(`confidence`), 그리고 어떤 체크포인트로
라우팅됐는지(`routing`, 한글/비영어는 자동으로 `multilingual` 체크포인트로 감)가 들어있다.

## 코드에서 직접 쓰기 (서버 없이)

```python
from laya import Router

router = Router()  # 첫 predict() 호출 때 필요한 체크포인트만 자동 다운로드
result = router.predict(state, questions)
result["answers"]["department"]["choice"]      # 예: "billing"
result["answers"]["urgency"]["score"]           # 예: 1.844 (0~2 범위)
```

## 질문 타입 3가지

- **choice**: 카테고리 중 하나 고르기 (`criteria`: {카테고리: 설명})
- **score**: 순서가 있는 등급 매기기 (`criteria`: 순서대로 나열한 리스트)
- **noul**: 예/아니오 확률 (`criteria`: {"false": ..., "true": ...})

## 확인한 것

- `uv add "laya[serve]"` 설치 → import/CLI 정상.
- `Router().predict(state, questions)` 직접 호출 → 한국어 입력을 자동으로
  `multilingual` 체크포인트로 라우팅, `department`/`urgency` 정상 응답.
- `uv run laya-serve` (`LAYA_PRELOAD=0`)로 실서버 기동 → `curl`로
  `/health`, `/v1/systemone` 둘 다 실제 HTTP로 호출해서 같은 결과 확인.
