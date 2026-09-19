# ============================================================================
# rl/vec_env.py — 여러 환경 인스턴스를 동시에 다루는 간단한 "벡터 환경"
#
# 왜 필요한가: PPO 같은 정책 경사(policy gradient) 계열 알고리즘은 한 번의
# 업데이트를 위해 "여러 에피소드에서 뽑은 다양한 경험"이 필요하다. 환경
# 하나만 순차적으로 돌리면, 그 환경이 우연히 처한 특정 상황(예: 항상 비슷한
# 방향으로 넘어짐)에 데이터가 치우쳐 학습이 불안정해지기 쉽다.
#
# 진짜 "병렬" 처리(멀티프로세싱)를 쓰면 더 빠르지만 구현이 복잡해지므로,
# 여기서는 파이썬 for문으로 N개의 환경을 순서대로 한 스텝씩 진행시키는
# "동기식(synchronous)" 벡터 환경을 쓴다. 물리 시뮬레이션 자체는 병렬로
# 빨라지지 않지만, 그래도 중요한 이점이 있다:
#   1) 신경망 순전파를 N개 환경의 관측값을 한꺼번에 배치(batch)로 넣어
#      한 번에 계산할 수 있어 효율적이다 (환경 하나씩 N번 호출하는 것보다 빠름).
#   2) 한 번의 PPO 업데이트에 N개의 서로 다른 상황(초기 자세, 목표 위치,
#      쓰러지는 타이밍 등)이 섞여 들어가 학습이 훨씬 안정적이다.
# ============================================================================

import numpy as np


class SyncVectorBipedEnv:
    def __init__(self, env_fns, reset_options=None):
        """reset_options: 모든 reset()(맨 처음뿐 아니라 에피소드가 끝나
        자동으로 다시 시작될 때도)에 매번 넘겨줄 옵션 딕셔너리. 예를 들어
        {"goal": (5.0, -1.0)}을 넘기면 모든 에피소드가 같은 목표로 고정된다
        (select_scenario.py로 고른 시나리오를 학습에 그대로 쓸 때 사용).
        """
        self.envs = [fn() for fn in env_fns]
        self.n = len(self.envs)
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.reset_options = reset_options

    def reset(self, seed=None):
        obs_list = []
        for i, env in enumerate(self.envs):
            s = None if seed is None else seed + i
            obs, _ = env.reset(seed=s, options=self.reset_options)
            obs_list.append(obs)
        return np.stack(obs_list)

    def step(self, actions):
        """actions: (N, action_dim) 배열. 각 환경에 하나씩 행동을 적용한다.

        환경 하나가 넘어지거나(terminated) 시간 초과(truncated)로 에피소드가
        끝나면, 그 환경만 즉시 reset()해서 다음 에피소드를 이어서 시작한다
        (한 환경이 먼저 끝났다고 나머지 N-1개를 멈추고 기다리지 않음 —
        이렇게 해야 N개 환경이 항상 쉬지 않고 데이터를 계속 생산한다).
        """
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for env, action in zip(self.envs, actions):
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if done:
                # PPO의 GAE(어드밴티지 계산)가 "에피소드가 여기서 끊겼다"를
                # 알 수 있도록 종료 직전 관측값을 info에 남겨두고,
                # 실제로 다음 스텝에 쓰일 obs는 새 에피소드의 시작값으로 교체한다.
                info = dict(info)
                info["terminal_observation"] = obs
                obs, _ = env.reset(options=self.reset_options)
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
        return (
            np.stack(obs_list),
            np.array(reward_list, dtype=np.float32),
            np.array(done_list, dtype=bool),
            info_list,
        )

    def close(self):
        for env in self.envs:
            env.close()
