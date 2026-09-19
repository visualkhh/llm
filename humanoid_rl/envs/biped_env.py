# ============================================================================
# envs/biped_env.py — 2족보행 로봇 강화학습 환경 (Gymnasium 인터페이스)
#
# 이 파일이 하는 일을 한 문장으로: robot/biped.urdf(순수 로봇 뼈대)를
# MuJoCo 물리엔진 위에 올리고, 거기에 (1) 지면 (2) 장애물 (3) 목표 지점을
# 덧붙여 "장애물을 피해 목표 지점으로 걸어가면 점수를 얻는" 하나의 완결된
# 강화학습 환경(Environment)으로 조립한다.
#
# 강화학습에서 "환경(Environment)"이 반드시 제공해야 하는 것은 표준적으로
# 아래 네 가지뿐이다 (Gymnasium 라이브러리가 정한 인터페이스):
#   reset()  : 에피소드를 처음부터 다시 시작하고 첫 관측값을 반환
#   step(a)  : 행동(a)을 한 스텝 실행하고 (다음 관측값, 보상, 종료여부, ...) 반환
#   observation_space : 관측값이 어떤 모양/범위의 숫자인지
#   action_space       : 에이전트가 낼 수 있는 행동이 어떤 모양/범위인지
#
# 이 네 가지만 정의해두면, rl/ppo.py 에 있는 PPO 알고리즘은 "이게 로봇인지
# 보드게임인지"조차 몰라도 학습을 진행할 수 있다 — 강화학습 알고리즘과
# 문제(환경)를 분리하는 것이 Gymnasium 표준의 핵심 아이디어다.
# ============================================================================

import os

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

from maps.generate_map import random_map_spec, SENSED_GEOM_GROUP

ROBOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF_PATH = os.path.join(ROBOT_DIR, "robot", "biped.urdf")
DEFAULT_MAP_PATH = os.path.join(ROBOT_DIR, "maps", "map_default.xml")

# --- 다리 관절 이름 (URDF에 정의된 순서와 정확히 일치해야 함) ---
LEG_JOINTS = [
    "l_hip_yaw_joint", "l_hip_roll_joint", "l_hip_pitch_joint", "l_knee_joint", "l_ankle_joint",
    "r_hip_yaw_joint", "r_hip_roll_joint", "r_hip_pitch_joint", "r_knee_joint", "r_ankle_joint",
]
# 각 관절의 최대 토크(Nm). URDF <limit effort="..."> 값과 맞춰, 액추에이터의
# ctrl 입력 범위 [-1, 1]이 그대로 "최대 토크 대비 몇 %"를 의미하게 만든다.
JOINT_GEARS = [60, 80, 120, 120, 80, 60, 80, 120, 120, 80]

# --- 라이다(거리 센서) 설정 ---
# 장애물의 정확한 world 좌표를 관측값에 그대로 꽂아주는 건 실제 로봇이라면
# 있을 수 없는 "치팅"이다 (실제 로봇은 자기 센서로 감지한 것만 알 수 있음).
# 그래서 장애물 좌표를 직접 주는 대신, 로봇 머리 위치에서 부채꼴로 광선을
# 여러 개 쏴서 "이 방향으로 몇 미터 앞에 뭔가 있다"만 알려주는 라이다 스타일
# 센서를 쓴다 (documents/08-lidar-sensor.md 참고).
N_LIDAR_RAYS = 15          # 부채꼴 안에서 쏘는 광선 개수 (해상도)
LIDAR_FOV_DEG = 120.0      # 정면 기준 시야각(도) — 예: 120도면 좌우 ±60도
LIDAR_MAX_RANGE = 6.0      # 이 거리(m)보다 먼 것/아무것도 없으면 그냥 최대값으로 채움
_LIDAR_GEOMGROUP = np.zeros(6, dtype=np.uint8)
_LIDAR_GEOMGROUP[SENSED_GEOM_GROUP] = 1  # 이 그룹(지면/장애물)만 감지, 로봇 자기 몸은 무시

STAND_HEIGHT = 0.9         # 두 발로 곧게 섰을 때 torso(골반) 중심의 대략적인 높이(m)
                            # (골반 하단 0.1 + 허벅지 0.35 + 정강이 0.35 + 발 여유 0.055 ≈ 0.855 + 여유)
FALL_HEIGHT = 0.6          # 이 높이 아래로 torso가 내려가면 "넘어짐"으로 판정
FALL_TILT_COS = 0.5        # torso의 "위쪽" 방향과 실제 월드 위쪽 방향의 코사인
                            # 유사도가 이 값보다 작아지면(=60도 이상 기울어지면) 넘어짐
GOAL_REACH_DIST = 0.5      # 목표까지 이 거리(m) 안으로 들어오면 "도착" 처리


def _quat_to_yaw(quat_wxyz):
    """torso의 회전(쿼터니언, w,x,y,z 순서)에서 '좌우로 얼마나 돌았는지(yaw)'만 뽑아낸다.

    사람이 몸을 좌우로 돌리는 각도(요, yaw)만 필요한 이유: 목표 지점이
    "내 앞 몇 미터, 왼쪽 몇 미터"인지를 계산하려면 로봇이 지금 어느 방향을
    보고 있는지를 알아야 하기 때문이다. (앞으로 넘어지거나 옆으로 기운
    정도(pitch, roll)는 별도의 '기울어짐' 관측값으로 이미 다루므로 여기선
    필요 없다.)
    """
    w, x, y, z = quat_wxyz
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _world_to_local(dx, dy, yaw):
    """월드 좌표계의 (dx, dy) 벡터를, 로봇이 바라보는 방향을 기준으로 한
    '로봇 기준 좌표계'로 회전 변환한다.

    이렇게 변환하는 이유: 로봇이 월드 좌표 (10, 3)에 있는 목표를 보든,
    (-5, 20)에 있는 목표를 보든, "내 정면 기준으로 몇 미터 앞, 몇 미터
    왼쪽에 있는가"라는 상대적인 정보는 로봇이 방향을 바꿀 때마다 자연스럽게
    갱신되고, 학습된 정책이 "월드의 절대 좌표"가 아니라 "지금 내 몸을
    기준으로 한 상황"에 반응하도록 만들어 훨씬 더 잘 일반화된다.
    (같은 상황이면 로봇이 어느 위치/방향에 있었든 같은 행동을 하면 되므로)
    """
    c, s = np.cos(-yaw), np.sin(-yaw)
    local_x = c * dx - s * dy
    local_y = s * dx + c * dy
    return local_x, local_y


class BipedEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode=None,
        max_episode_steps=1000,
        frame_skip=5,
        map_path=None,
        randomize_map=False,
        goal_range=(3.0, 6.0),
    ):
        """
        frame_skip        : 에이전트가 행동 1개를 고를 때마다, 물리엔진은
                             내부적으로 이 횟수만큼 더 잘게 시간을 쪼개 계산한다.
                             MuJoCo의 기본 물리 시간간격(timestep)은 0.002초로
                             매우 세밀한데, 에이전트가 매 0.002초마다 새로운
                             판단을 내릴 필요는 없고(계산 낭비) 그렇게 하면
                             학습도 훨씬 느려지므로, frame_skip=5로 묶어
                             "에이전트는 0.01초(=0.002*5)마다 한 번씩 행동을
                             고른다"로 만든다.
        map_path           : 불러올 맵(.xml, MJCF) 파일 경로. None이면
                             maps/map_default.xml 사용. maps/generate_map.py로
                             만든 파일이나 사용자가 직접 작성한 맵 파일도
                             그대로 넣을 수 있다 (documents/06-map-files.md 참고).
        randomize_map      : True면 map_path를 무시하고, 에피소드(reset)마다
                             매번 새로운 무작위 장애물 배치를 즉석에서 생성해
                             사용한다 (파일로 저장하지 않고 메모리에서만 생성 —
                             MuJoCo의 spec 재컴파일이 1ms 미만으로 매우 저렴해서
                             매 에피소드 다시 만들어도 학습 속도에 지장이 없다).
        goal_range         : 매 에피소드 시작 시 목표 지점을 로봇 앞
                             (goal_range[0] ~ goal_range[1]) 미터, 좌우
                             ±2미터 범위에서 무작위로 뽑음.
        """
        super().__init__()
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.frame_skip = frame_skip
        self.map_path = map_path or DEFAULT_MAP_PATH
        self.randomize_map = randomize_map
        self.goal_range = goal_range

        self._rebuild_world(rng=np.random.default_rng())

        n_joints = len(LEG_JOINTS)
        # 행동: 8개 관절 각각에 대해 -1~1 사이의 값 하나씩 -> 실제 토크로 환산됨
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n_joints,), dtype=np.float32)

        obs_dim = (
            n_joints          # 관절 각도
            + n_joints        # 관절 각속도
            + 1                # torso 기울어짐 정도(코사인 유사도, 1=완벽히 수직)
            + 3                # torso 선속도
            + 3                # torso 각속도
            + 2                # 목표까지의 상대 위치(로봇 기준 좌표계, x=앞, y=왼쪽)
            + 1                # 목표까지의 거리
            + N_LIDAR_RAYS     # 라이다 광선별 감지 거리 (장애물 좌표를 직접 주지 않음)
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._step_count = 0
        self._goal_xy = np.zeros(2)
        self._prev_dist_to_goal = 0.0
        self._renderer = None

    # ------------------------------------------------------------------
    # 모델 조립: 맵(.xml, 지면+장애물) + URDF(순수 로봇) + 목표 마커 + 액추에이터
    #
    # "맵"과 "로봇"이 완전히 독립된 두 개의 MjSpec으로 시작해서, attach()로
    # 하나의 시뮬레이션 모델로 합쳐지는 구조다. 이렇게 분리해두면:
    #   - 로봇은 그대로 두고 맵만 바꿔가며 다양한 지형에서 실험할 수 있고,
    #   - 반대로 맵은 그대로 두고 로봇만 바꿔볼 수도 있다.
    # 자세한 배경은 documents/06-map-files.md 참고.
    # ------------------------------------------------------------------
    def _build_model(self, rng):
        if self.randomize_map:
            # 파일로 저장하지 않고, 그때그때 무작위 장애물 배치를 메모리에서 생성
            map_spec = random_map_spec(rng)
        else:
            map_spec = mujoco.MjSpec.from_file(self.map_path)

        map_spec.option.timestep = 0.002
        # 접촉이 많은 다리/발 시뮬레이션은 기본 solver 반복 횟수를 늘려야
        # 발이 지면을 뚫고 들어가는 등의 불안정한 현상이 줄어든다.
        map_spec.option.iterations = 50

        # --- 로봇을 맵의 원점(0,0,0)에 붙인다 ---
        # attach()는 기본적으로 이름 충돌 방지를 위해 "/" 접두사를 붙이는데,
        # 이 프로젝트는 로봇이 하나뿐이라 접두사 없이(prefix="") 붙여서
        # LEG_JOINTS에 정의한 이름을 그대로 쓸 수 있게 한다.
        robot_spec = mujoco.MjSpec.from_file(URDF_PATH)
        spawn_frame = map_spec.worldbody.add_frame()
        map_spec.attach(robot_spec, frame=spawn_frame, prefix="")

        # --- 목표 지점 마커: mocap body로 만든다 ---
        # 처음엔 "site"로 만들고 매번 model.site_pos만 바꾸면 될 줄
        # 알았는데, 실제로 해보니 site_pos는 컴파일 시점 값으로 취급되어
        # mj_forward를 다시 불러도 data.site_xpos(실제 렌더링에 쓰이는 값)에
        # 반영되지 않는 것을 확인했다. MuJoCo에서 "물리 법칙의 영향은 받지
        # 않지만 매 스텝 외부에서 위치를 갱신하고 싶은 오브젝트"를 위한
        # 정식 매커니즘이 바로 mocap body이며, data.mocap_pos를 갱신하면
        # mj_forward가 매번 정확히 반영해준다.
        goal_body = map_spec.worldbody.add_body(name="goal_marker", mocap=True)
        goal_body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            # size = [반지름, 반높이, 미사용]. 반높이를 작게 둬서 바닥에 놓인
            # 납작한 원반처럼 보이게 함.
            size=[0.3, 0.02, 0],
            rgba=[0.1, 0.7, 0.9, 0.85],
            contype=0,      # 로봇/장애물과 물리적으로 충돌하지 않는 순수 표시용
            conaffinity=0,
        )

        # --- 액추에이터: URDF에도, 맵 파일에도 없던 "모터"를 여기서 붙임 ---
        # (URDF를 순수하게 유지하고 제어 방식은 여기서 붙이는 이유는
        #  documents/01-urdf-and-mujoco.md 참고)
        for jname, gear in zip(LEG_JOINTS, JOINT_GEARS):
            map_spec.add_actuator(
                name=f"{jname}_motor",
                target=jname,
                trntype=mujoco.mjtTrn.mjTRN_JOINT,
                gainprm=[gear] + [0] * 9,   # motor 액추에이터: gain*ctrl = 토크
                ctrllimited=True,
                ctrlrange=[-1.0, 1.0],
            )

        model = map_spec.compile()

        # 각 다리 관절이 qpos/qvel 배열에서 몇 번째 인덱스인지 미리 찾아둔다.
        # (root_joint가 자유 조인트라 qpos 7칸/qvel 6칸을 차지하고, 이제는
        #  waist/neck/어깨/팔꿈치 같은 수동 관절도 끼어 있어서, 다리 관절이
        #  정확히 몇 번째 칸인지 하드코딩으로는 알 수 없다 — 반드시 MuJoCo
        #  API로 물어봐야 하며, 이렇게 해두면 URDF에 관절을 추가/순서변경해도
        #  코드가 깨지지 않는다)
        leg_joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in LEG_JOINTS]
        qpos_adr = [model.jnt_qposadr[jid] for jid in leg_joint_ids]
        qvel_adr = [model.jnt_dofadr[jid] for jid in leg_joint_ids]

        # 맵 파일에 들어있는 장애물들을 "obstacle_0, obstacle_1, ..." 이름
        # 규칙으로 순서대로 찾는다 (maps/generate_map.py 의 스키마 참고).
        # 이름이 끊기는 지점에서 멈추므로, 장애물이 몇 개든 자동으로 인식된다.
        obstacle_xy = []
        i = 0
        while True:
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle_{i}")
            if gid == -1:
                break
            obstacle_xy.append((float(model.geom_pos[gid][0]), float(model.geom_pos[gid][1])))
            i += 1

        goal_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "goal_marker")
        goal_mocap_id = model.body_mocapid[goal_body_id]
        return model, qpos_adr, qvel_adr, obstacle_xy, goal_mocap_id

    def _rebuild_world(self, rng):
        """맵+로봇을 (다시) 조립하고, 관련 캐시 값들을 갱신한다.

        고정된 맵을 쓸 때는 __init__에서 딱 한 번만 호출되지만,
        randomize_map=True일 때는 매 reset()마다 호출되어 매번 새로운
        지형을 만든다 (재컴파일 비용이 1ms 미만이라 성능에 지장이 없음).
        """
        self.model, self._joint_qpos_adr, self._joint_qvel_adr, self._obstacle_xy, self._goal_mocap_id = \
            self._build_model(rng)
        self.data = mujoco.MjData(self.model)
        self._torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        # 라이다 광선의 발사 지점으로 쓸 "머리" 위치 (센서 마운트 지점)
        self._head_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "head")

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}

        if self.randomize_map:
            # 매 에피소드 새 지형: 모델 전체를 새로 조립 (mj_resetData로는
            # 부족함 — 장애물 개수/위치 자체가 바뀌므로 재컴파일이 필요)
            self._rebuild_world(rng=self.np_random)
        else:
            mujoco.mj_resetData(self.model, self.data)

        # torso를 두 발로 선 자세 근처에서 시작시킨다 (완전히 처음부터
        # "일어서는 법"까지 배우게 하면 문제가 훨씬 더 어려워지므로,
        # 첫 버전에서는 "이미 서 있는 상태"에서 "걷기+목표도달"만 학습).
        self.data.qpos[0:3] = [0.0, 0.0, STAND_HEIGHT]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # 회전 없음(단위 쿼터니언)

        # 매번 완전히 똑같은 자세에서 시작하면 정책이 "그 자세 하나"만 외울
        # 위험이 있으므로, 관절 각도에 아주 작은 무작위 잡음을 섞어 다양한
        # 시작 상태에서도 안정적으로 대응하도록 만든다.
        rng = self.np_random
        noise = rng.uniform(-0.03, 0.03, size=len(LEG_JOINTS))
        for adr, n in zip(self._joint_qpos_adr, noise):
            self.data.qpos[adr] = n

        # 목표 지점: 기본은 로봇 앞쪽(goal_range 거리, 좌우 무작위)에 무작위로 배치.
        # options={'goal': (x, y)}로 넘기면 사용자가 원하는 특정 지점으로 고정 가능
        # (요청하신 "목표 지점을 직접 찍어주는" 기능이 바로 이 부분).
        if "goal" in options:
            gx, gy = options["goal"]
        else:
            dist = rng.uniform(*self.goal_range)
            angle = rng.uniform(-0.6, 0.6)  # 정면 기준 좌우 약 ±34도 이내
            gx, gy = dist * np.cos(angle), dist * np.sin(angle)
        self._goal_xy = np.array([gx, gy])
        self.data.mocap_pos[self._goal_mocap_id] = [gx, gy, 0.02]

        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._prev_dist_to_goal = float(np.linalg.norm(self._goal_xy - self.data.qpos[0:2]))

        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action

        # frame_skip번 물리 시뮬레이션을 진행 (한 번의 "에이전트 행동"이
        # 실제로는 이만큼의 짧은 물리 스텝 동안 유지되며 힘을 가한다는 뜻)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        obs = self._get_obs()

        torso_xy = self.data.qpos[0:2].copy()
        torso_z = float(self.data.qpos[2])
        dist_to_goal = float(np.linalg.norm(self._goal_xy - torso_xy))
        tilt_cos = self._tilt_cos()

        fell = (torso_z < FALL_HEIGHT) or (tilt_cos < FALL_TILT_COS)
        reached_goal = dist_to_goal < GOAL_REACH_DIST

        reward, reward_info = self._compute_reward(
            dist_to_goal=dist_to_goal, action=action, fell=fell, reached_goal=reached_goal
        )
        self._prev_dist_to_goal = dist_to_goal

        terminated = bool(fell or reached_goal)
        truncated = self._step_count >= self.max_episode_steps

        info = {"reached_goal": reached_goal, "fell": fell, "dist_to_goal": dist_to_goal}
        info.update(reward_info)
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def _tilt_cos(self):
        """torso가 얼마나 수직으로 서 있는지를 0~1 사이 값 하나로 요약.

        data.xmat[torso_id]는 torso 좌표축이 월드 좌표계에서 어떻게 회전해
        있는지를 나타내는 3x3 회전행렬(9개 원소로 펼쳐짐)이다. 그중 마지막
        열(인덱스 8)이 'torso의 z축(위쪽)이 월드 z축과 얼마나 나란한지'
        (내적, 코사인 유사도)를 바로 알려준다. 완전히 똑바로 서 있으면 1,
        옆으로 90도 누우면 0, 거꾸로 뒤집히면 -1에 가까워진다.
        """
        xmat = self.data.xmat[self._torso_id]
        return float(xmat[8])

    def _lidar_scan(self, yaw):
        """머리 위치에서 정면 기준 ±(LIDAR_FOV_DEG/2)도 부채꼴로 광선
        N_LIDAR_RAYS개를 쏴서, 각 방향으로 가장 가까운 장애물/지면까지의
        거리를 잰다. 장애물의 정확한 좌표를 직접 알려주는 대신, 실제
        로봇의 거리 센서(라이다)처럼 "이 방향엔 몇 미터 앞에 뭔가 있다"만
        알 수 있게 하기 위함이다 (documents/08-lidar-sensor.md 참고).

        아무것도 안 걸리면(-1이 반환되면) "그 방향은 비어있다"는 뜻으로
        최댓값(LIDAR_MAX_RANGE)을 채운다.
        """
        origin = self.data.xpos[self._head_id].copy()
        half_fov = np.radians(LIDAR_FOV_DEG) / 2.0
        angles = yaw + np.linspace(-half_fov, half_fov, N_LIDAR_RAYS)

        geomid = np.zeros(1, dtype=np.int32)
        dists = np.empty(N_LIDAR_RAYS, dtype=np.float32)
        for i, angle in enumerate(angles):
            vec = np.array([np.cos(angle), np.sin(angle), 0.0])
            d = mujoco.mj_ray(self.model, self.data, origin, vec, _LIDAR_GEOMGROUP, 1, -1, geomid)
            dists[i] = LIDAR_MAX_RANGE if d < 0 else min(float(d), LIDAR_MAX_RANGE)
        return dists

    def _get_obs(self):
        joint_pos = np.array([self.data.qpos[a] for a in self._joint_qpos_adr])
        joint_vel = np.array([self.data.qvel[a] for a in self._joint_qvel_adr])

        yaw = _quat_to_yaw(self.data.qpos[3:7])
        torso_xy = self.data.qpos[0:2]

        gx_local, gy_local = _world_to_local(*(self._goal_xy - torso_xy), yaw)
        goal_dist = float(np.linalg.norm(self._goal_xy - torso_xy))

        lidar = self._lidar_scan(yaw)

        obs = np.concatenate([
            joint_pos,
            joint_vel,
            [self._tilt_cos()],
            self.data.qvel[0:3],
            self.data.qvel[3:6],
            [gx_local, gy_local],
            [goal_dist],
            lidar,
        ]).astype(np.float32)
        return obs

    def _compute_reward(self, dist_to_goal, action, fell, reached_goal):
        # 1) 전진 보상: "지난 스텝보다 목표에 얼마나 가까워졌는가"
        #    거리 자체가 아니라 '거리의 변화량'을 보상으로 주는 이유는,
        #    거리 자체를 매 스텝 그대로 보상(혹은 음의 보상)으로 주면 목표에서
        #    멀리 시작한 에피소드와 가까이 시작한 에피소드의 보상 스케일이
        #    달라져 학습이 불안정해지기 때문. '변화량'은 항상 비슷한 스케일을 가짐.
        progress = self._prev_dist_to_goal - dist_to_goal
        progress_reward = 10.0 * progress

        # 2) 생존 보너스: 매 스텝 살아있는 것만으로 약간의 보상 -> 일부러
        #    빨리 넘어져 에피소드를 끝내버리는 '자포자기' 전략을 막아줌
        alive_bonus = 0.5

        # 3) 제어 비용: 관절에 너무 큰 힘을 계속 쓰는 것에 작은 페널티
        #    -> 불필요하게 격렬하게 팔다리를 휘두르는 대신 효율적인 동작 선호
        ctrl_cost = 0.01 * float(np.sum(np.square(action)))

        reward = progress_reward + alive_bonus - ctrl_cost

        # 4) 목표 도달 / 넘어짐: 에피소드를 결정짓는 큰 보상/페널티
        if reached_goal:
            reward += 50.0
        if fell:
            reward -= 20.0

        info = {
            "reward_progress": progress_reward,
            "reward_alive": alive_bonus,
            "reward_ctrl_cost": -ctrl_cost,
        }
        return reward, info

    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data, camera="track" if self._has_track_cam() else -1)
            return self._renderer.render()
        return None

    def _has_track_cam(self):
        return False

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
