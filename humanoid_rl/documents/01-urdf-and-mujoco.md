# 01. URDF와 MuJoCo — 로봇을 어떻게 "정의"하고 "제어"하는가

관련 코드: `robot/biped.urdf`, `envs/biped_env.py`의 `_build_model`

## URDF란

URDF(Unified Robot Description Format)는 ROS(Robot Operating System)
생태계에서 널리 쓰이는 XML 기반 로봇 기술(記述) 포맷입니다. 로봇을
"링크(link, 뼈대 조각)"와 "조인트(joint, 링크 사이를 잇는 관절)"의 트리
구조로 표현합니다.

```
<link>   : 질량, 관성, 모양(시각적/충돌용)을 가진 강체(rigid body) 하나
<joint>  : 두 링크를 연결하는 관절. type으로 움직임의 종류를 정함
             - fixed    : 안 움직임 (완전히 고정)
             - revolute : 한 축을 중심으로 제한된 각도만큼 회전(경첩)
             - floating : (MuJoCo 확장) 완전히 자유로운 6자유도 부유
```

이 프로젝트의 로봇(`biped.urdf`)은 다음과 같은 트리로 되어 있습니다
(자세한 그림은 URDF 파일 맨 위 주석 참고):

```
world → torso(floating, 골반) → chest(revolute, waist) → head(revolute, neck)
                                                        → l/r_shoulder_roll → l/r_shoulder_pitch → l/r_lower_arm(elbow)  [모터 없음=수동]
                              → l/r_hip_yaw → l/r_hip_roll → l/r_hip_pitch
                                   → l/r_shin(knee) → l/r_foot(ankle)   [다리 5관절×2=10개, 강화학습이 제어]
```

waist/neck/shoulder/elbow는 관절이 있어 물리적으로 움직이지만
(`envs/biped_env.py`에서 모터를 붙이지 않으므로) 강화학습이 직접 제어하는
대상은 아니다 — 걷는 동안 관성/충격으로 자연스럽게 흔들리는 수동 관절.

## 왜 URDF에는 "모터"가 없는가

URDF는 원래 "이 로봇이 물리적으로 어떻게 생겼는가"만 기술하는 포맷이고,
"어떤 소프트웨어/제어기로 어떻게 움직일 것인가"는 별도 영역(ROS에서는
`ros_control`의 `<transmission>` 태그 등)입니다. MuJoCo도 마찬가지로
액추에이터(모터)는 원칙적으로 MJCF(MuJoCo의 네이티브 XML)의 `<actuator>`
섹션에 정의하도록 되어 있고, 순수 URDF 문법 자체에는 그런 섹션이 없습니다.

실험해본 결과(이 문서 작성 과정에서 직접 확인):
- URDF 안에 `<mujoco><compiler .../></mujoco>` 확장 블록은 인식되지만,
- 그 안에 `<actuator>`를 넣어도 무시됨 (MuJoCo의 URDF 파서는 compiler
  옵션 정도만 확장으로 허용하고, 모델 구조 확장은 지원하지 않음)

그래서 이 프로젝트는 **URDF는 순수하게 로봇 자체의 정의로 유지**하고,
모터처럼 "이 로봇을 가지고 무엇을 할 것인가"에 속하는 부분은 파이썬
코드(`envs/biped_env.py`)에서 MuJoCo의 **`MjSpec`** API로 프로그래밍하듯이
덧붙이는 방식을 택했습니다. 지면/장애물/목표 지점은 한 걸음 더 나아가
URDF도 아니고 파이썬 코드도 아닌, **완전히 별도의 맵(.xml) 파일**로
분리했습니다 (자세한 내용은 [06-map-files.md](06-map-files.md) 참고).

```python
map_spec = mujoco.MjSpec.from_file("maps/map_default.xml")  # 지면+장애물 (독립 파일)
robot_spec = mujoco.MjSpec.from_file("biped.urdf")           # 순수 로봇 구조

frame = map_spec.worldbody.add_frame()
map_spec.attach(robot_spec, frame=frame, prefix="")   # 로봇을 맵의 원점에 붙임
map_spec.worldbody.add_site(...)                       # 목표 마커 추가 (충돌 없음)
map_spec.add_actuator(target="l_hip_yaw_joint", ...)   # 관절에 모터 붙이기
model = map_spec.compile()                             # 최종적으로 하나의 시뮬레이션 모델로 컴파일
```

이 방식의 장점: 로봇 정의(URDF), 맵 정의(MJCF), 제어 방식(파이썬의
액추에이터 설정)이 세 개의 독립된 조각으로 분리되어 있어서, **로봇은
그대로 두고 맵 파일만 바꿔 끼우거나, 맵은 그대로 두고 로봇만 바꿔보는**
식의 실험이 쉽습니다.

## `type="floating"` — URDF의 표준 밖 확장

표준 URDF 조인트 타입에는 `floating`이 없습니다 (`fixed`, `revolute`,
`continuous`, `prismatic`, `planar` 가 표준). 하지만 로봇 전체가 공중에
뜬 채로 자유롭게 움직이고 넘어질 수도 있어야 하는 이족보행 로봇 같은
경우, "몸통이 세상(world)에 대해 완전히 자유롭게 움직인다"는 것을
표현해야 하는데, 표준 조인트 타입만으로는 이걸 표현할 방법이 없습니다.

실험으로 확인한 결과 MuJoCo의 URDF 파서는 `type="floating"`을 인식해서
내부적으로 MJCF의 `freejoint`(6자유도: 위치 3 + 회전 4를 쿼터니언으로,
총 `qpos` 7칸/`qvel` 6칸)로 변환해줍니다. 이게 `world → torso` 사이에
있는 `root_joint`의 정체이며, 이 덕분에 로봇 몸통이 넘어지거나 점프하거나
자유롭게 이동할 수 있습니다.

## 좌표계와 qpos/qvel 인덱스

URDF/ROS 좌표계 규칙: **x=정면, y=왼쪽, z=위**.

`root_joint`(floating)가 맨 앞이라 MuJoCo의 상태 벡터는 이렇게 구성됩니다:

```
qpos[0:3]   = torso(골반) 위치 (x, y, z)
qpos[3:7]   = torso 회전 (쿼터니언, w, x, y, z 순서)
qpos[7:15]  = waist, neck, l_shoulder_roll, l_shoulder_pitch, l_elbow,
              r_shoulder_roll, r_shoulder_pitch, r_elbow (수동 관절 8개)
qpos[15:25] = 10개 다리 관절 각도
              (l_hip_yaw, l_hip_roll, l_hip_pitch, l_knee, l_ankle,
               r_hip_yaw, r_hip_roll, r_hip_pitch, r_knee, r_ankle)

qvel[0:3]   = torso 선속도 (월드 좌표계 기준)
qvel[3:6]   = torso 각속도 (torso 로컬 좌표계 기준 — MuJoCo의 관례)
qvel[6:14]  = 수동 관절 8개의 각속도
qvel[14:24] = 10개 다리 관절 각속도
```

`envs/biped_env.py`는 이 인덱스를 하드코딩하지 않고 `model.jnt_qposadr`/
`model.jnt_dofadr`로 "이 관절 이름이 실제로 몇 번째 칸에 있는지"를 안전하게
물어봅니다 — 처음엔 다리 관절 속도를 `qvel[6:14]`로 하드코딩했었는데,
waist/neck/어깨/팔꿈치 관절을 추가하면서 실제 인덱스가 밀려버려 엉뚱한
값을 읽는 버그가 났었다. 이름으로 안전하게 조회하도록 고친 뒤로는 URDF에
관절을 추가/순서 변경해도 파이썬 코드가 깨지지 않는다.

## 관절을 2개로 쪼갠 이유 (hip_yaw + hip_pitch)

사람의 고관절(엉덩이 관절)은 공처럼 여러 방향으로 회전하는 볼 조인트지만,
URDF의 `revolute` 조인트는 **회전축 하나(1자유도)** 만 표현할 수 있습니다.
그래서 "좌우 회전(yaw)"과 "앞뒤 흔들기(pitch)" 두 가지 자유도를 주려면
그 사이에 질량이 거의 없는 가상의 중간 링크(`l_hip_yaw_link`)를 하나
끼워, 두 개의 1자유도 조인트를 연달아 연결하는 방식을 씁니다. 이 중간
링크는 실체가 있는 부위가 아니라 순수하게 기구학적 연결을 위한 트릭입니다.

## 관성값(inertia)은 어떻게 정했나

정밀한 인체측정 데이터 대신, 각 신체 부위를 단순한 상자(box)나 구(sphere)로
근사해서 표준 공식으로 계산했습니다.

```
상자(가로 x, 세로 y, 높이 z, 질량 m):
  Ixx = m/12 * (y² + z²)
  Iyy = m/12 * (x² + z²)
  Izz = m/12 * (x² + y²)

구(반지름 r, 질량 m):
  Ixx = Iyy = Izz = (2/5) * m * r²
```

교육/실험용으로는 충분히 그럴듯한 물리적 거동을 만들어내지만, 실제
로봇 설계에 쓸 수 있는 정밀도는 아닙니다.

## 수동(passive) 관절 — waist / neck / shoulder / elbow

다리의 10개 관절과 달리, 허리·목·어깨(2축)·팔꿈치에는 `envs/biped_env.py`에서
모터(액추에이터)를 붙이지 않습니다. 대신 URDF의 `<joint>` 안에
`<dynamics damping="..." friction="..."/>`를 넣어두었는데, 이건 그
관절에 "가상의 마찰/저항"을 주는 설정입니다.

```xml
<joint name="l_shoulder_pitch_joint" type="revolute">
  ...
  <dynamics damping="1.5" friction="0.05"/>
</joint>
```

`damping`이 없으면 그 관절은 어떤 저항도 없이 자유롭게 빙글빙글 도는
경첩과 같아서, 걷다가 받은 충격만으로도 팔이나 머리가 비현실적으로
계속 흔들리거나 통제 불능으로 튈 수 있습니다. `damping` 값을 주면
"움직이는 속도에 비례해서 반대 방향으로 힘이 걸리는" 저항이 생겨,
관성으로 한 번 흔들린 뒤 서서히 가라앉는 자연스러운 움직임이 됩니다
(자동차 서스펜션의 댐퍼와 같은 원리). `friction`은 그 외에 아주 작은
정지 마찰을 더해 완전히 멈췄을 때 미세하게 계속 떨리는 것을 막습니다.

이 방식의 장점: 로봇이 여전히 "관절이 있고 그 관절이 물리적으로
움직이는" 사실적인 몸을 갖게 되면서도, 강화학습의 행동 공간(현재 8개
다리 액추에이터)은 그대로 유지되어 학습 문제의 난이도가 커지지
않습니다. 나중에 이 관절들도 직접 제어하고 싶다면, `envs/biped_env.py`의
`LEG_JOINTS`/`JOINT_GEARS` 목록에 이름과 토크값을 추가하고
`observation_space` 크기만 맞춰주면 됩니다 (다만 행동 공간이 커질수록
학습이 어려워진다는 점은 감안해야 합니다).
