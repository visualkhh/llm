# 07. 시나리오 선택과 실시간 학습 시각화

관련 코드: `select_scenario.py`, `rl/live_view.py`, `train.py`의 `--scenario`/`--watch`

## 전체 흐름

```
1) select_scenario.py 실행
     -> maps/ 폴더에서 맵을 고르거나 무작위로 하나 생성
     -> 뷰어가 열리고, 로봇 앞에 초록색 원반(목표 후보)이 보임
     -> 방향키로 원반을 원하는 위치로 옮기고 Enter로 확정
     -> scenario.json 파일로 저장됨: {"map_path": ..., "goal": [x, y]}

2) uv run python train.py --scenario scenario.json --watch
     -> 저장된 맵/목표를 그대로 쓰면서 학습 시작
     -> 모든 병렬 환경(N개)을 한 창에 동시에 실시간으로 보여줌
```

## 왜 마우스 클릭이 아니라 방향키인가

앞서 실험해본 결과, MuJoCo의 공식 인터랙티브 뷰어(`mujoco.viewer.launch_passive`)는
`key_callback`(키보드)만 공식적으로 지원하고, "화면의 어느 픽셀을 클릭했는지를
3D 월드 좌표로 변환"하는 기능(레이캐스팅)은 공식 API에 없습니다. 이걸
직접 구현하려면 MuJoCo의 저수준 렌더링 함수(`mjv_*`, `mjr_*`)와 GLFW를
직접 조합해 카메라의 투영행렬로부터 클릭 지점을 3D 광선으로 변환하고,
그 광선이 지면(z=0)과 만나는 점을 계산하는 별도의 렌더링 파이프라인을
새로 짜야 합니다. 지금은 공식적으로 지원되는 키보드 콜백으로 커서를
움직이는 방식을 택했고, 진짜 마우스 클릭이 꼭 필요하면 이후 별도
작업으로 진행할 수 있습니다.

## select_scenario.py 사용법

> **macOS 주의:** `select_scenario.py`와 `play.py`는 인터랙티브 3D
> 창(`mujoco.viewer.launch_passive`)을 띄우는데, macOS는 GUI 창을 반드시
> "메인 스레드"에서만 만들 수 있게 강제해서, 일반 `python`이 아니라
> MuJoCo가 제공하는 **`mjpython`**으로 실행해야 합니다 (그냥 `python`으로
> 실행하면 `RuntimeError: launch_passive requires ... mjpython` 에러가 남).
> `uv`로 설치했다면 `.venv/bin/mjpython`이 이미 들어있으므로 `uv run
> mjpython ...`처럼 `python` 대신 `mjpython`만 바꿔주면 됩니다.
> (반면 `train.py --watch`는 화면에 안 보이는 offscreen 렌더링 +
> OpenCV 창을 쓰기 때문에 이 제약이 없고, 그냥 `python`/`uv run python`으로
> 실행하면 됩니다.)

```bash
uv run mjpython select_scenario.py                        # 대화형으로 맵 선택
uv run mjpython select_scenario.py --map_path maps/map_default.xml   # 맵 선택 생략하고 바로 시작
```

방향키(↑↓←→) 또는 W/A/S/D로 청록색 원반(목표)을 움직이고, Enter로
확정합니다. 로봇은 이 화면에서는 물리 시뮬레이션이 진행되지 않고 정지된
자세로 고정되어 있습니다 (목표를 고르는 동안 넘어지면 산만하므로
일부러 멈춰둠).

> **참고 (겪었던 버그):** 처음 구현에서는 목표 마커를 MuJoCo의 `site`로
> 만들고 매 프레임 `model.site_pos`만 갱신했는데, `site_pos`는 컴파일
> 시점 값으로 취급되어 `mj_forward`를 다시 불러도 실제 렌더링 좌표에
> 반영되지 않아 "키를 눌러도 화면에서 안 움직이는" 문제가 있었습니다.
> 지금은 목표 마커를 **mocap body**로 만들어(`data.mocap_pos`를
> 갱신하면 `mj_forward`가 항상 정확히 반영해줌) 해결했습니다
> (`envs/biped_env.py`의 `_build_model` 참고).

확정하면 `scenario.json`에 다음과 같이 저장됩니다:

```json
{"map_path": "maps/map_default.xml", "goal": [4.25, -0.75]}
```

## train.py / play.py에서 시나리오 쓰기

```bash
uv run python train.py --scenario scenario.json --watch
uv run mjpython play.py --scenario scenario.json
```

`--scenario`를 주면 `--map_path`/무작위 목표 설정을 모두 무시하고, 파일에
저장된 맵과 목표로 **모든 에피소드가 고정**됩니다 (매 에피소드 다른
무작위 목표로 학습하는 기본 동작과 달리, "이 지점 하나로 정확히 갈 수
있는지"를 집중적으로 학습/확인하고 싶을 때 유용합니다).

## 실시간 다중 환경 시각화 (`--watch`)

```bash
uv run python train.py --watch                       # 기본 8개 환경을 4x2 격자로
uv run python train.py --watch --watch_cols 2         # 2열로 배치
uv run python train.py --watch --watch_fps 30         # 화면 갱신 빈도(재생 속도) 조절
```

### 왜 MuJoCo 뷰어 창을 8개 띄우지 않았는가

MuJoCo의 인터랙티브 뷰어는 원래 "사람이 카메라를 돌려가며 하나의 시뮬레이션을
자세히 들여다보는 것"을 위해 설계되어 있어서, 창 하나당 모델(시뮬레이션)
하나만 다룹니다. 8개를 한 번에 보려고 창을 8개 띄우면 창 관리도 번거롭고
자원 소모도 커집니다. 대신 각 환경을 화면에 띄우지 않고 메모리 안에서만
렌더링(`mujoco.Renderer`, offscreen rendering)해서 이미지로 뽑아낸 뒤,
그 이미지들을 격자로 이어붙여 OpenCV 창 하나로 보여주는 방식을 택했습니다
(`rl/live_view.py`의 `LiveGridViewer`).

### 학습 속도에 미치는 영향

렌더링은 물리 스텝보다 훨씬 비싼 연산이라, 매 스텝 8개를 전부 다시 그리면
학습이 눈에 띄게 느려집니다. 그래서 `LiveGridViewer`는 내부적으로
"1초에 `fps_cap`번까지만 실제로 화면을 갱신"하도록 스스로 프레임을
건너뜁니다(기본 20fps). 사람 눈에는 초당 20장이면 충분히 매끄럽게 보이고,
학습 자체는 그 사이에도 방해받지 않고 계속 진행됩니다. 그래도 `--watch`
없이 돌릴 때보다는 어느 정도 느려지므로, 정확한 학습 속도가 중요할 때는
`--watch` 없이 돌리는 것을 권장합니다.

### 화면에서 쓸 수 있는 조작

| 키 | 동작 |
|---|---|
| `space` | 일시정지 / 재개 (화면 갱신만 멈춤, 학습 자체는 계속 진행됨) |
| `+` / `-` | 화면 갱신 fps 조절 (10~30 사이에서 취향껏) |
| `n` | env[0]의 맵을 "매 에피소드 무작위 생성"으로 즉시 전환 (다이나믹 맵 교체) |
| `q` | 시각화 창만 닫음 — 학습은 백그라운드에서 그대로 계속됨 |

`n`을 누르면 화면 왼쪽 위 첫 번째 칸에 보이는 환경이 그 순간부터 매
에피소드 새로운 무작위 장애물 배치를 쓰기 시작합니다. 이 기능은
`BipedEnv.randomize_map` 플래그를 실행 중에 켜는 방식으로 구현되어
있으며, 필요하면 `rl/live_view.py`의 `on_key` 콜백을 확장해 다른 환경도
같은 방식으로 조작할 수 있습니다.

### 화면 상단 통계

이터레이션 번호, 누적 스텝 수, 최근 에피소드 평균 보상/길이, 목표 도달
성공률(%), 엔트로피, KL, value loss가 매 PPO 업데이트마다 갱신되어
화면 위쪽에 표시됩니다 — 터미널 로그를 계속 들여다보지 않아도 학습이
잘 되고 있는지 화면만 보고 바로 알 수 있습니다.
