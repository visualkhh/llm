# 06. 맵 파일 — 로봇과 분리된 "세상" 정의

관련 코드: `maps/generate_map.py`, `maps/map_default.xml`, `envs/biped_env.py`의 `_build_model`

## USD가 아니라 MJCF인 이유

"맵/씬 파일" 하면 USD(Universal Scene Description)를 떠올릴 수 있는데,
USD는 Pixar가 만들고 NVIDIA Omniverse/Isaac Sim 생태계가 널리 쓰는
포맷입니다. 이 프로젝트는 시뮬레이터로 **MuJoCo**를 쓰고 있고, MuJoCo는
USD를 읽거나 쓰지 못합니다 (완전히 다른 생태계). MuJoCo에서 로봇에게
URDF가 있듯, "세상(씬)"에 해당하는 MuJoCo의 표준 포맷은 **MJCF(.xml)**
입니다. 그래서 이 프로젝트의 맵 파일도 MJCF로 되어 있습니다.

(만약 정말 USD/Isaac Sim 생태계로 가고 싶다면, 그건 시뮬레이터 자체를
MuJoCo에서 Isaac Sim으로 바꾸는 훨씬 큰 결정이 됩니다 — 로봇 URDF는
어느 정도 재사용 가능하지만, 물리엔진/제어 코드는 다시 짜야 합니다.)

## 맵과 로봇을 왜 분리했는가

`robot/biped.urdf`가 "로봇 그 자체"만 담당하듯, `maps/*.xml`은 "로봇이
그 위에서 움직일 세상"만 담당합니다. 이렇게 분리하면:

- 로봇은 그대로 두고 맵만 바꿔가며 다양한 지형/장애물 배치에서 실험 가능
- 맵 파일이 사람이 읽을 수 있는 평범한 XML 텍스트라서, 코드를 몰라도
  텍스트 에디터로 장애물을 추가/이동/삭제할 수 있음
- 나중에 로봇을 다른 종류로 바꾸더라도 맵 파일은 그대로 재사용 가능

## 맵 파일의 스키마(규칙)

```xml
<mujoco model="map">
  <worldbody>
    <geom name="ground" type="plane" size="30 30 0.1" group="1" .../>
    <geom name="obstacle_0" type="box" pos="2 0.6 0.8" size="0.2 0.2 0.8" group="1" .../>
    <geom name="obstacle_1" type="box" pos="3.5 -0.7 0.8" size="0.2 0.2 0.8" group="1" .../>
    <!-- obstacle_2, obstacle_3, ... 원하는 만큼 추가 가능 -->
  </worldbody>
</mujoco>
```

`envs/biped_env.py`가 맵을 해석할 때 지키는 규칙은 세 가지입니다:

1. 지면 geom의 이름은 `ground` (있어도 되고 없어도 되지만, 없으면 로봇이
   허공에 서 있는 맵이 됨 — 특별한 의도가 아니라면 항상 넣을 것)
2. 장애물 geom은 이름이 `obstacle_0`, `obstacle_1`, `obstacle_2`, ...
   처럼 0부터 끊기지 않고 이어져야 함. 환경 코드가 `obstacle_0`부터
   순서대로 이름을 찾다가, 존재하지 않는 번호가 나오면 그 지점에서 멈춥니다.
3. **지면/장애물 geom에는 `group="1"`을 넣어야 함.** 로봇의 라이다(거리
   센서)가 "이 그룹만 감지한다"는 필터를 쓰고 있어서([08-lidar-sensor.md](08-lidar-sensor.md)),
   이걸 빠뜨리면 로봇이 물리적으로는 부딪히면서도 센서로는 못 보는
   "투명한" 장애물이 되어버립니다. `maps/generate_map.py`로 만들면
   자동으로 붙지만, 손으로 직접 작성/수정할 때는 꼭 챙겨야 합니다.

이 세 가지만 지키면, geom의 모양(box/sphere/cylinder), 크기, 색깔, 개수는
전부 자유입니다. 로봇은 항상 맵의 월드 원점 `(0, 0)` 근처에 스폰되므로,
장애물은 그 원점을 기준으로 배치하면 됩니다.

## 세 가지 사용 방법

### 1) 기본 맵 그대로 쓰기

```python
env = BipedEnv()  # maps/map_default.xml 자동 사용
```

### 2) 맵을 무작위로 생성해서 파일로 저장 (직접 열어서 편집 가능)

```bash
uv run python maps/generate_map.py --out maps/random_01.xml --seed 3 --n_min 2 --n_max 5
```

이렇게 생성된 파일은 평범한 텍스트 XML이라 바로 열어서 장애물 위치나
크기를 손으로 고칠 수 있습니다. 고친 뒤 그대로 다시 불러오면 됩니다:

```python
env = BipedEnv(map_path="maps/random_01.xml")
```

```bash
uv run python train.py --map_path maps/random_01.xml
uv run mjpython play.py  --map_path maps/random_01.xml
```

### 3) 매 에피소드 완전히 새로운 무작위 맵 (학습 중 지형 다양화) — `train.py`의 기본 동작

```python
env = BipedEnv(randomize_map=True)
```

```bash
uv run python train.py                    # map_path/scenario를 안 주면 이게 기본값
uv run python train.py --randomize_map    # 명시적으로 켜고 싶을 때
```

이 경우 파일을 저장하지 않고, `maps/generate_map.py`의 `random_map_spec()`
함수로 그때그때 메모리에서 새 장애물 배치를 만들어 씁니다. MuJoCo의 모델
재컴파일이 1ms도 안 걸릴 만큼 가벼워서(직접 측정 결과) 매 에피소드
다시 만들어도 학습 속도에 지장이 없습니다. 이렇게 학습하면 정책이 특정
장애물 배치 하나를 외우는 대신, 장애물을 "인식하고 피하는" 좀 더 일반적인
행동을 배우도록 강제할 수 있습니다.

**`train.py`는 `--map_path`나 `--scenario`를 따로 지정하지 않으면 이
무작위화를 기본으로 사용합니다.** 고정된 맵 하나로만 학습시키면 학습이
끝난 뒤 `play.py`에 다른 맵을 넣어봤을 때 제대로 대응하지 못하기
쉽습니다 — "장애물을 피한다"는 행동을 일반화해서 배운 게 아니라, 그
맵 하나의 배치를 외운 것에 가까워지기 때문입니다(이러면 강화학습이라기보다
그 상황 하나에 맞춘 선언적인 해법에 가까워짐). 정말 고정 맵 하나로만
빠르게 맛보기 학습을 해보고 싶다면 `--fixed_map`을 명시적으로 줘야 합니다:

```bash
uv run python train.py --fixed_map    # 디버깅/맛보기용. 다른 맵 일반화는 기대하지 말 것
```

## 사용자가 맵을 직접 만드는 법

`maps/map_default.xml`을 복사해서 `<geom name="obstacle_N" .../>` 줄을
원하는 대로 추가/수정하면 됩니다. MJCF 문법 전체를 몰라도 이 정도
패턴만 따라 하면 충분합니다. 원한다면 장애물 모양을 상자(`box`) 대신
`sphere`, `cylinder` 등으로 바꿔도 되고(둘 다 MuJoCo가 지원하는 기본
geom 타입), 색깔(`rgba`)이나 크기(`size`)도 자유롭게 바꿀 수 있습니다.

## 파이썬 코드로 맵을 만드는 법 (파일 저장 없이)

```python
from maps.generate_map import build_map_spec

my_obstacles = [
    (2.0, 0.0, 0.15, 0.15, 0.6),   # (x, y, half_x, half_y, height)
    (4.0, 1.0, 0.3, 0.3, 1.0),
]
spec = build_map_spec(my_obstacles)   # 필요하면 spec.compile(); spec.to_file(...) 로 저장도 가능
```
