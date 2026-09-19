# ============================================================================
# rl/ppo.py — PPO(Proximal Policy Optimization) 알고리즘 직접 구현 (TensorFlow)
#
# 강화학습이 풀려는 문제를 한 문장으로 요약하면:
#   "지금 상태(observation)를 보고 어떤 행동(action)을 해야 미래에 받을
#    보상(reward)의 총합이 최대가 되는가?"를 시행착오로 배우는 것.
#
# 이 파일은 그 중에서도 정책 경사(Policy Gradient) 계열의 대표 알고리즘인
# PPO를 구현한다. 큰 그림은 아래 세 부분으로 나뉜다.
#   1) ActorCritic  — "지금 상태에서 어떤 행동이 좋을까"(Actor)와
#                      "지금 상태가 앞으로 얼마나 좋은 상태인가"(Critic)를
#                      동시에 추정하는 신경망
#   2) GAE          — 실제로 얻은 보상들로부터 "이 행동이 얼마나 잘한
#                      행동이었는지"(advantage, 이점)를 계산하는 방법
#   3) PPO 업데이트  — advantage를 이용해 Actor를 개선하되, 한 번에 너무
#                      크게 바뀌지 않도록 "제약을 걸어(Proximal)" 안정적으로
#                      학습하는 방법
#
# 각 개념의 자세한 배경은 documents/03-ppo-algorithm.md 를 함께 참고할 것.
# ============================================================================

import numpy as np
import tensorflow as tf
from tensorflow import keras


def _mlp(input_dim, hidden_sizes, output_dim, output_activation=None, name=None):
    """단순한 완전연결(fully-connected) 신경망 하나를 만드는 헬퍼.

    hidden_sizes 개수만큼 은닉층을 쌓고, 각 은닉층 뒤에는 tanh 활성함수를 둔다
    (강화학습에서는 ReLU보다 tanh가 값의 범위를 -1~1로 눌러줘 정책/가치
    추정이 조금 더 안정적으로 학습되는 경향이 있어 관례적으로 많이 쓰인다).
    """
    inputs = keras.Input(shape=(input_dim,))
    x = inputs
    for h in hidden_sizes:
        x = keras.layers.Dense(h, activation="tanh")(x)
    outputs = keras.layers.Dense(output_dim, activation=output_activation)(x)
    return keras.Model(inputs, outputs, name=name)


class ActorCritic:
    """Actor(정책 네트워크) + Critic(가치 네트워크)을 함께 들고 있는 컨테이너.

    - Actor: 상태(obs) -> 행동의 평균(mean). 실제 행동은 이 평균을 중심으로 한
      정규분포(Normal distribution)에서 "무작위로 샘플링"해서 정해진다.
      왜 확률적으로 뽑는가: 학습 초반에는 어떤 행동이 좋은지 전혀 모르므로,
      다양한 행동을 시도해봐야(탐험, exploration) 무엇이 좋은지 알 수 있다.
      학습이 진행될수록 표준편차(std)가 스스로 줄어들며 점점 더 확신을 갖고
      행동하게 된다.
    - Critic: 상태(obs) -> 그 상태의 가치(value) 추정치 하나(스칼라).
      "이 상태에서 시작해 앞으로 얻을 보상의 합이 대략 얼마일까"를 예측한다.
      이 값은 GAE에서 "실제로 얻은 보상이 예상보다 좋았는지 나빴는지"를
      비교하는 기준선(baseline)으로 쓰인다.

    log_std를 상태와 무관한 하나의 학습 가능한 벡터로 둔 이유(널리 쓰이는
    관례): 매 상태마다 별도로 탐험 정도를 계산하게 하면 학습이 불안정해지기
    쉬운 반면, "지금 이 관절은 평균적으로 얼마나 확신을 갖고 움직이는가"를
    상태와 무관하게 하나로 두면 학습이 더 안정적이면서도 충분히 잘 작동한다.
    """

    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256)):
        self.act_dim = act_dim
        self.actor = _mlp(obs_dim, hidden_sizes, act_dim, output_activation="tanh", name="actor")
        self.critic = _mlp(obs_dim, hidden_sizes, 1, name="critic")
        # log(표준편차)를 직접 학습 대상으로 둔다 (표준편차 자체를 학습시키면
        # 음수가 되지 않도록 항상 조심해야 하므로, log값을 학습시키고 필요할
        # 때 exp를 취하는 것이 표준적인 트릭).
        self.log_std = tf.Variable(np.full(act_dim, -0.5, dtype=np.float32), trainable=True)

    @property
    def trainable_variables(self):
        return self.actor.trainable_variables + self.critic.trainable_variables + [self.log_std]

    def get_action_and_value(self, obs, action=None):
        """obs가 주어졌을 때:
          - action이 None이면 정책에서 새 행동을 '샘플링'해서 반환 (환경과
            상호작용할 때, 즉 데이터를 모을 때 쓰는 경로)
          - action이 주어지면, 그 행동을 '현재' 정책 기준으로 다시 평가만
            해서 log-확률/엔트로피를 반환 (PPO 업데이트 때, 예전에 모아둔
            행동을 지금 정책 기준으로 재평가할 때 쓰는 경로)
        두 경로를 하나의 함수로 합쳐두면 아래 rollout 수집과 PPO 업데이트
        코드에서 정책 평가 로직을 중복 작성하지 않아도 된다.
        """
        mean = self.actor(obs)
        log_std = tf.clip_by_value(self.log_std, -5.0, 2.0)  # 극단적인 값으로 발산 방지
        std = tf.exp(log_std)

        if action is None:
            noise = tf.random.normal(shape=tf.shape(mean))
            action = mean + noise * std

        # 정규분포의 log-확률 밀도 함수를 직접 계산 (여러 독립적인 관절 행동의
        # 결합 확률이므로, 각 차원의 log-확률을 더한다 = 독립사건 확률의 곱의 로그)
        var = tf.square(std)
        log_prob = -0.5 * (
            tf.reduce_sum(tf.square(action - mean) / var, axis=-1)
            + tf.reduce_sum(tf.math.log(2.0 * np.pi * var), axis=-1)
        )
        # 엔트로피: 지금 정책이 얼마나 "무작위/탐험적"인지의 척도. PPO 손실에
        # 작은 보너스로 더해 정책이 너무 일찍 한 가지 행동에만 확신을 갖고
        # 굳어버리는 것(조기 수렴)을 막는다.
        entropy = tf.reduce_sum(log_std + 0.5 * np.log(2.0 * np.pi * np.e), axis=-1)
        # entropy는 배치 전체에서 같은 std를 공유하므로 스칼라 하나를 배치 크기로 broadcast
        entropy = tf.fill(tf.shape(log_prob), entropy) if entropy.shape.rank == 0 else entropy

        value = tf.squeeze(self.critic(obs), axis=-1)
        return action, log_prob, entropy, value

    def get_value(self, obs):
        return tf.squeeze(self.critic(obs), axis=-1)


def compute_gae(rewards, values, dones, last_values, gamma=0.99, lam=0.95):
    """GAE(Generalized Advantage Estimation) — "이 행동이 평균보다 얼마나
    좋았는가(advantage)"를 계산한다.

    입력은 모두 (T, N) 모양 — T=수집한 타임스텝 수, N=병렬 환경 개수.
    values, last_values는 Critic이 예측한 가치 추정치.

    핵심 아이디어(TD residual, delta): 한 스텝의 "예상과 실제의 차이"는
        delta_t = (r_t + gamma * V(s_{t+1})) - V(s_t)
    로 정의된다. "한 스텝 뒤 상황까지 봤을 때 다시 계산한 가치"에서
    "원래 이 상태에서 예측했던 가치"를 뺀 것 — 양수면 "생각보다 좋은 결과",
    음수면 "생각보다 나쁜 결과"였다는 뜻이다.

    GAE는 이 delta들을 미래 방향으로 gamma*lam 비율로 감쇠시키며 누적한다
    (아래 for 루프를 뒤에서부터, 즉 시간 역순으로 도는 이유가 이것 —
    각 시점의 advantage는 "그 뒤에 일어난 delta들의 감쇠합"이기 때문에
    미래의 계산 결과(lastgaelam)를 이용해 과거로 거슬러 올라가며 계산하는
    것이 가장 효율적이다).

    lam(λ)=0이면 한 스텝만 보는 이점 추정(분산은 작지만 편향이 큼), λ=1이면
    에피소드 끝까지 모든 보상을 그대로 반영하는 몬테카를로 추정(편향은
    없지만 분산이 큼)에 가까워진다. λ=0.95는 그 사이의 실용적인 절충값으로
    널리 쓰인다.

    returns(=advantage + value)는 Critic을 학습시킬 때의 "정답"으로 쓰인다.
    """
    T, N = rewards.shape
    advantages = np.zeros_like(rewards)
    last_gae_lam = np.zeros(N, dtype=np.float32)

    for t in reversed(range(T)):
        if t == T - 1:
            next_value = last_values
        else:
            next_value = values[t + 1]
        next_nonterminal = 1.0 - dones[t]

        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae_lam = delta + gamma * lam * next_nonterminal * last_gae_lam
        advantages[t] = last_gae_lam

    returns = advantages + values
    return advantages, returns


class PPO:
    def __init__(
        self,
        obs_dim,
        act_dim,
        lr=3e-4,
        clip_ratio=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
        train_epochs=10,
        minibatch_size=256,
    ):
        self.ac = ActorCritic(obs_dim, act_dim)
        self.optimizer = keras.optimizers.Adam(learning_rate=lr)
        self.clip_ratio = clip_ratio
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.train_epochs = train_epochs
        self.minibatch_size = minibatch_size

        # --- 파이프라이닝(pipeline)을 위한 "추론 전용 사본" ---
        # train.py --pipeline 모드에서는 "다음 롤아웃 수집"과 "지금 데이터로
        # 학습"을 서로 다른 스레드에서 동시에 진행한다. 이때 두 스레드가
        # 정확히 같은 가중치(self.ac)를 동시에 읽고(추론) 쓰면(경사하강)
        # 경쟁 상태(race condition)가 생길 수 있다 — 업데이트가 절반쯤
        # 끝난 애매한 상태의 가중치로 행동을 결정해버릴 수도 있다는 뜻.
        #
        # 이를 안전하게 피하기 위해 "학습되는 진짜 가중치(ac)"와 "롤아웃
        # 수집에만 쓰는, 잠시 얼려둔 사본(ac_infer)"을 분리한다. 롤아웃
        # 수집 스레드는 ac_infer만 읽고, 학습 스레드는 ac만 갱신하므로 절대
        # 같은 메모리를 동시에 건드리지 않는다. 매 이터레이션이 끝난
        # 뒤(두 스레드가 모두 끝난 안전한 시점) `sync_infer()`로 방금 학습한
        # 최신 가중치를 ac_infer에 복사해, 다음 라운드 롤아웃부터는 새
        # 정책을 쓰게 한다 — 그래서 "한 이터레이션만큼 살짝 뒤처진(stale)"
        # 정책으로 수집한다는 점만 빼면 학습에 미치는 영향은 미미하다.
        self.ac_infer = ActorCritic(obs_dim, act_dim)
        self.sync_infer()

        # tf.function으로 감싸서 "그래프 모드"로 컴파일해둔다. 기본(eager) 모드는
        # 한 줄 한 줄을 파이썬이 그때그때 해석하며 실행해서 이해하기는 쉽지만
        # 느리다. tf.function은 처음 호출될 때 딱 한 번만 계산 그래프를
        # 컴파일해두고, 그 다음부터는 파이썬 인터프리터를 거치지 않고 컴파일된
        # 그래프를 그대로 실행한다 — 특히 이 프로젝트처럼 아주 작은 신경망을
        # 아주 많이(한 이터레이션에 수십~수백 번) 호출하는 경우, 파이썬 자체의
        # 호출 오버헤드 비중이 커서 tf.function 하나로 학습 속도가 크게
        # 개선된다(직접 측정 결과, 대략 2~4배).
        self._act_fn = tf.function(self.ac_infer.get_action_and_value)
        self._value_fn = tf.function(self.ac_infer.get_value)
        self._train_step_fn = tf.function(self._train_step)

    def sync_infer(self):
        """방금 학습한 최신 가중치(self.ac)를 롤아웃 수집용 사본(ac_infer)에
        복사한다. train.py의 파이프라인 루프가 매 이터레이션 경계(두 스레드가
        모두 끝난 뒤)에 호출해서, 다음 롤아웃부터 최신 정책을 쓰게 한다.
        """
        self.ac_infer.actor.set_weights(self.ac.actor.get_weights())
        self.ac_infer.critic.set_weights(self.ac.critic.get_weights())
        self.ac_infer.log_std.assign(self.ac.log_std)

    def act(self, obs):
        """환경과 상호작용할 때 쓰는 진입점. obs: (N, obs_dim) numpy 배열."""
        obs_tf = tf.convert_to_tensor(obs, dtype=tf.float32)
        action, log_prob, _, value = self._act_fn(obs_tf)
        return action.numpy(), log_prob.numpy(), value.numpy()

    def value(self, obs):
        obs_tf = tf.convert_to_tensor(obs, dtype=tf.float32)
        return self._value_fn(obs_tf).numpy()

    def update(self, obs, actions, old_log_probs, advantages, returns):
        """수집한 rollout 데이터로 여러 epoch에 걸쳐 미니배치 단위 업데이트를 수행.

        obs, actions, old_log_probs, advantages, returns: 모두 이미
        (T*N, ...) 형태로 펼쳐진(flatten) 배열이라고 가정한다.

        같은 데이터를 train_epochs번 반복해서 쓰는 이유: 환경에서 새 데이터를
        모으는 것(시뮬레이션 실행)이 신경망 업데이트보다 훨씬 비싸기 때문에,
        한 번 모은 데이터를 여러 번 재사용해 학습 효율을 높인다. 다만 같은
        데이터를 너무 많이/크게 재사용하면 정책이 "그 데이터를 모을 때의
        정책"에서 너무 멀어져 버릴 수 있는데, 이를 막는 장치가 바로 아래
        PPO의 clip(자르기)이다.
        """
        n = obs.shape[0]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        old_log_probs = tf.convert_to_tensor(old_log_probs, dtype=tf.float32)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)

        stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_frac": []}

        for _ in range(self.train_epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, self.minibatch_size):
                mb_idx = idx[start:start + self.minibatch_size]
                policy_loss, value_loss, entropy_loss, approx_kl, clip_frac = self._train_step_fn(
                    tf.gather(obs, mb_idx),
                    tf.gather(actions, mb_idx),
                    tf.gather(old_log_probs, mb_idx),
                    tf.gather(advantages, mb_idx),
                    tf.gather(returns, mb_idx),
                )
                stats["policy_loss"].append(float(policy_loss))
                stats["value_loss"].append(float(value_loss))
                stats["entropy"].append(float(-entropy_loss))
                stats["approx_kl"].append(float(approx_kl))
                stats["clip_frac"].append(float(clip_frac))

        return {k: float(np.mean(v)) for k, v in stats.items()}

    def _train_step(self, mb_obs, mb_actions, mb_old_logp, mb_adv, mb_ret):
        """미니배치 하나에 대한 PPO 업데이트 한 스텝. `update()`가 이 함수를
        `tf.function`으로 감싸 반복 호출한다 (그래프 모드로 컴파일해 속도 향상).
        """
        with tf.GradientTape() as tape:
            _, new_log_probs, entropy, values = self.ac.get_action_and_value(mb_obs, action=mb_actions)

            # ratio = 새 정책이 이 행동을 택할 확률 / 예전 정책이 택했을 확률
            # (로그 확률의 차를 지수화 = 확률의 비율)
            ratio = tf.exp(new_log_probs - mb_old_logp)

            # --- PPO의 핵심: Clipped Surrogate Objective ---
            # ratio가 1보다 훨씬 커지거나 작아지면(=정책이 그 사이 너무
            # 많이 바뀌었다는 뜻) advantage에 곱해지는 비율을
            # [1-clip_ratio, 1+clip_ratio] 범위로 강제로 잘라버린다.
            # min을 취함으로써 "advantage가 양수인데 ratio가 너무 커진
            # 경우"뿐 아니라 "advantage가 음수인데 ratio가 너무 작아진
            # 경우"까지 모두 안전하게 보수적으로(pessimistic) 처리한다.
            surr1 = ratio * mb_adv
            surr2 = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * mb_adv
            policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

            # Critic 손실: 예측한 가치(values)가 실제 계산된 returns에
            # 가까워지도록 하는 평범한 회귀(regression) 문제
            value_loss = tf.reduce_mean(tf.square(values - mb_ret))

            entropy_loss = -tf.reduce_mean(entropy)

            loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

        grads = tape.gradient(loss, self.ac.trainable_variables)
        # gradient clipping: 한 번의 업데이트에서 가중치가 너무 크게
        # 튀는 것을 막아 학습을 안정시킨다 (특히 정책이 예상 밖의
        # 상황을 만났을 때 손실이 순간적으로 커질 수 있는 RL에서 중요)
        grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        self.optimizer.apply_gradients(zip(grads, self.ac.trainable_variables))

        approx_kl = tf.reduce_mean(mb_old_logp - new_log_probs)
        clip_frac = tf.reduce_mean(tf.cast(tf.abs(ratio - 1.0) > self.clip_ratio, tf.float32))
        return policy_loss, value_loss, entropy_loss, approx_kl, clip_frac
