# ============================================================================
# train.py — 여러 환경을 동시에 돌리며 PPO로 2족보행 로봇을 학습시키는 스크립트
#
# 한 번의 "학습 이터레이션"이 하는 일:
#   1) N개의 환경에서 각각 T스텝씩, 총 N*T개의 (상태, 행동, 보상, ...) 데이터를 수집
#      (rl/vec_env.py) — 이 동안은 신경망 파라미터를 전혀 바꾸지 않고 "지금
#      정책 그대로" 데이터만 모은다.
#   2) 모은 데이터로 GAE를 계산해 각 행동이 "얼마나 좋았는지"(advantage)를 구함
#   3) 그 advantage를 이용해 PPO 손실로 Actor/Critic을 여러 epoch 업데이트
#   4) 1~3을 total_timesteps에 도달할 때까지 반복
#
# 자세한 배경 지식은 documents/ 폴더, 특히 03-ppo-algorithm.md 와
# 04-training-loop.md 를 참고.
# ============================================================================

import argparse
import json
import os
import threading
import time

import numpy as np

from envs.biped_env import BipedEnv
from rl.vec_env import SyncVectorBipedEnv
from rl.ppo import PPO, compute_gae

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


def make_env(map_path=None, randomize_map=False):
    return lambda: BipedEnv(
        max_episode_steps=1000, frame_skip=5, map_path=map_path, randomize_map=randomize_map
    )


def make_env_kwargs(map_path=None, randomize_map=False):
    return dict(max_episode_steps=1000, frame_skip=5, map_path=map_path, randomize_map=randomize_map)


def save_checkpoint(agent, ckpt_name):
    agent.ac.actor.save_weights(os.path.join(CKPT_DIR, f"{ckpt_name}_actor.weights.h5"))
    agent.ac.critic.save_weights(os.path.join(CKPT_DIR, f"{ckpt_name}_critic.weights.h5"))
    np.save(os.path.join(CKPT_DIR, f"{ckpt_name}_log_std.npy"), agent.ac.log_std.numpy())


def collect_rollout(
    vec_env, agent, obs, n_steps, n_envs, obs_dim, act_dim,
    live_viewer_box, last_stats_box,
    ep_returns, ep_lengths, finished_returns, finished_lengths, finished_success,
):
    """n_steps만큼 n_envs개 환경을 진행시켜 롤아웃 버퍼를 채운다.

    --pipeline 모드에서는 이 함수가 백그라운드 스레드에서 실행되며,
    agent.act()/agent.value()는 내부적으로 agent.ac_infer(학습 스레드가
    건드리지 않는 "얼려둔" 사본)만 사용하므로 메인 스레드에서 동시에 진행되는
    agent.update()(agent.ac를 갱신)와 가중치를 두고 경합하지 않는다.
    """
    buf_obs = np.zeros((n_steps, n_envs, obs_dim), dtype=np.float32)
    buf_actions = np.zeros((n_steps, n_envs, act_dim), dtype=np.float32)
    buf_log_probs = np.zeros((n_steps, n_envs), dtype=np.float32)
    buf_values = np.zeros((n_steps, n_envs), dtype=np.float32)
    buf_rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
    buf_dones = np.zeros((n_steps, n_envs), dtype=np.float32)

    for t in range(n_steps):
        actions, log_probs, values = agent.act(obs)
        buf_obs[t] = obs
        buf_actions[t] = actions
        buf_log_probs[t] = log_probs
        buf_values[t] = values

        obs, rewards, dones, infos = vec_env.step(actions)
        buf_rewards[t] = rewards
        buf_dones[t] = dones.astype(np.float32)

        if live_viewer_box[0] is not None:
            still_watching = live_viewer_box[0].maybe_update(last_stats_box[0])
            if not still_watching:
                live_viewer_box[0].close()
                live_viewer_box[0] = None  # 창만 닫힘. 학습 루프는 그대로 계속 진행됨

        ep_returns += rewards
        ep_lengths += 1
        for i, done in enumerate(dones):
            if done:
                finished_returns.append(ep_returns[i])
                finished_lengths.append(ep_lengths[i])
                finished_success.append(1.0 if infos[i].get("reached_goal") else 0.0)
                ep_returns[i] = 0.0
                ep_lengths[i] = 0

    last_values = agent.value(obs)
    return buf_obs, buf_actions, buf_log_probs, buf_values, buf_rewards, buf_dones, last_values, obs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_envs", type=int, default=8, help="동시에 돌릴 환경 개수")
    parser.add_argument("--n_steps", type=int, default=256, help="환경 하나당 한 이터레이션에 모을 스텝 수")
    parser.add_argument("--total_timesteps", type=int, default=2_000_000,
                         help="총 학습 스텝 수. 실제 이터레이션 횟수는 "
                              "total_timesteps // (n_envs * n_steps)로 자동 계산된다 "
                              "(예: 기본값 기준 2,000,000 // (8*256) = 976회). 더 오래/많이 "
                              "학습시키고 싶으면 이 값을 늘리면 된다.")
    parser.add_argument("--lr", type=float, default=3e-4,
                         help="학습률(learning rate). 한 번의 업데이트에서 가중치를 얼마나 크게 "
                              "이동시킬지 정한다. 너무 크면 loss가 진동/발산하고, 너무 작으면 "
                              "학습이 지나치게 느려진다. 3e-4는 Transformer/PPO 계열에서 경험적으로 "
                              "잘 통하는 값이라 기본값으로 씀 (documents/03-ppo-algorithm.md 참고).")
    parser.add_argument("--gamma", type=float, default=0.99,
                         help="할인율(discount factor). GAE에서 '당장의 보상'과 '먼 미래의 보상' "
                              "중 얼마나 미래를 중요하게 볼지 정한다. 1에 가까울수록 미래를 더 "
                              "중시한다 (documents/03-ppo-algorithm.md의 GAE 절 참고).")
    parser.add_argument("--lam", type=float, default=0.95,
                         help="GAE의 lambda 값. advantage 추정의 편향(bias)과 분산(variance) 사이 "
                              "절충을 정한다. 0이면 분산은 작지만 Critic 오차에 편향되고, 1이면 "
                              "편향은 없지만 들쭉날쭉(분산 큼)해진다. 0.95는 실무에서 널리 쓰이는 "
                              "절충값.")
    parser.add_argument("--clip_ratio", type=float, default=0.2,
                         help="PPO의 clip 범위(epsilon). 한 번의 업데이트로 정책이 예전 정책 대비 "
                              "[1-clip_ratio, 1+clip_ratio] 범위를 넘어서 바뀌지 못하게 막는 "
                              "안전장치. 기본값 0.2가 PPO 논문 및 대부분의 구현에서 쓰이는 표준값.")
    # epochs=4, minibatch=512: --multiprocess --pipeline과 함께 쓸 때 업데이트
    # 자체를 더 짧게 만들어 롤아웃 수집과 겹치는 비율을 높인다 (실측: 초당
    # 이터레이션 약 20% 향상). documents/04-training-loop.md의 "파이프라이닝"
    # 절 참고.
    parser.add_argument("--train_epochs", type=int, default=4,
                         help="한 번 모은 롤아웃 데이터를 몇 번(epoch) 재사용해서 업데이트할지. "
                              "크게 하면 데이터를 더 알뜰하게 쓰지만(샘플 효율↑) 업데이트 자체가 "
                              "오래 걸리고, 너무 크면 정책이 그 데이터를 모을 때의 정책에서 너무 "
                              "멀어질 위험도 커진다(clip이 어느 정도 막아주긴 함). 기본값 4는 "
                              "--pipeline과 궁합이 좋도록(업데이트를 짧게) 낮춰둔 값 — 예전 "
                              "표준값인 10으로 되돌리면 데이터를 더 우려먹는 대신 업데이트가 "
                              "느려진다.")
    parser.add_argument("--minibatch_size", type=int, default=512,
                         help="한 번의 경사하강 스텝에 쓰는 표본 개수. 크게 하면 미니배치 개수가 "
                              "줄어 업데이트가 빨라지지만(파이썬/TF 호출 오버헤드 감소), 그래디언트 "
                              "추정이 덜 세밀해질 수 있다. n_steps*n_envs(한 이터레이션에 모으는 "
                              "총 표본 수)보다 커지면 미니배치가 하나로 합쳐진다.")
    parser.add_argument("--save_interval", type=int, default=10, help="몇 이터레이션마다 체크포인트를 저장할지")
    parser.add_argument("--ckpt_name", type=str, default="ppo_biped",
                         help="체크포인트 파일 이름의 접두사. checkpoints/{ckpt_name}_actor.weights.h5 "
                              "등으로 저장되며, play.py --ckpt_name으로 불러올 때도 이 이름을 맞춰줘야 "
                              "한다. 서로 다른 실험을 같은 checkpoints/ 폴더에 저장하면서 구분하고 "
                              "싶을 때 바꿔주면 된다 (예: --ckpt_name exp1).")
    parser.add_argument("--map_path", type=str, default=None,
                         help="사용할 맵(.xml) 파일 경로를 고정. 주지 않으면(그리고 --scenario도 "
                              "없으면) 기본적으로 --randomize_map처럼 매 에피소드 무작위 맵으로 학습한다.")
    parser.add_argument("--fixed_map", action="store_true",
                         help="map_path/scenario를 안 줘도 무작위화하지 않고 maps/map_default.xml "
                              "하나로 고정 학습 (쉬운 맛보기/디버깅용. 이렇게 학습하면 그 맵 하나만 "
                              "잘 풀 뿐 다른 맵에서는 잘 안 될 수 있음 — 결국 '하나의 정답을 외운' "
                              "것에 가까워 진짜 일반화된 정책이라 보기 어렵다).")
    parser.add_argument("--randomize_map", action="store_true",
                         help="(기본 동작과 동일, 명시적으로 켜고 싶을 때) 매 에피소드 무작위 장애물 배치")
    parser.add_argument("--scenario", type=str, default=None,
                         help="select_scenario.py로 만든 scenario.json 경로. "
                              "주어지면 map_path/goal을 이 파일 내용으로 고정한다.")
    parser.add_argument("--watch", action="store_true",
                         help="학습 중인 모든 환경을 하나의 창에 실시간으로 시각화")
    parser.add_argument("--watch_fps", type=int, default=20,
                         help="--watch 창의 초당 화면 갱신 횟수(fps). 실제 학습 속도와는 무관하고 "
                              "'얼마나 자주 다시 그릴지'만 정한다. 렌더링이 물리 스텝보다 비싸서, "
                              "너무 높이면(예: 60) 그만큼 학습 속도가 느려질 수 있다. 창 안에서도 "
                              "+/- 키로 실시간 조절 가능.")
    parser.add_argument("--watch_cols", type=int, default=4,
                         help="--watch 창에서 환경들을 몇 열(column)의 격자로 배치할지. n_envs=8, "
                              "watch_cols=4면 4x2 격자가 된다.")
    parser.add_argument("--multiprocess", action="store_true",
                         help="환경마다 별도 OS 프로세스를 띄워 여러 CPU 코어에서 진짜 병렬로 "
                              "실행한다 (놀고 있는 코어를 활용해 롤아웃 수집 속도를 높임). "
                              "--watch와는 함께 쓸 수 없다 (각 환경이 다른 프로세스에 있어서 "
                              "메인 프로세스에서 화면을 직접 그릴 수 없음).")
    parser.add_argument("--pipeline", action="store_true",
                         help="'다음 롤아웃 수집'과 '지금 데이터로 PPO 업데이트'를 동시에(백그라운드 "
                              "스레드로) 진행해서, 신경망 학습 중 환경(워커)이 노는 시간을 없앤다. "
                              "--multiprocess와 함께 쓸 때 효과가 가장 크다. --watch와는 함께 쓸 수 "
                              "없다(화면 그리기는 메인 스레드에서만 안전함). 한 이터레이션만큼 살짝 "
                              "뒤처진(stale) 정책으로 롤아웃을 수집하게 되는 부작용이 있다.")
    args = parser.parse_args()

    if args.multiprocess and args.watch:
        raise SystemExit(
            "--multiprocess와 --watch는 함께 쓸 수 없습니다 (각 환경이 별도 프로세스에 있어 "
            "실시간 렌더링을 할 수 없음). 시각화가 필요하면 --watch만, 속도가 필요하면 "
            "--multiprocess만 쓰세요."
        )
    if args.pipeline and args.watch:
        raise SystemExit(
            "--pipeline과 --watch는 함께 쓸 수 없습니다 (백그라운드 스레드에서 화면을 그리면 "
            "macOS 등에서 불안정할 수 있음). 시각화가 필요하면 --watch만, 속도가 필요하면 "
            "--pipeline만 쓰세요."
        )

    os.makedirs(CKPT_DIR, exist_ok=True)

    reset_options = None
    map_path = args.map_path
    if args.scenario:
        with open(args.scenario) as f:
            scenario = json.load(f)
        map_path = scenario["map_path"]
        reset_options = {"goal": tuple(scenario["goal"])}
        print(f"시나리오 로드: map={map_path}  goal={reset_options['goal']}")

    # 맵 무작위화 기본 정책: map_path나 scenario로 "이 맵을 쓰겠다"고 명시하지
    # 않았고 --fixed_map도 안 줬다면, 기본적으로 무작위 맵으로 학습한다.
    # 고정된 맵 하나로만 학습하면 그 배치 하나를 외우는 것에 가까워, 정작
    # 학습이 끝난 뒤 다른 맵을 넣어보면 잘 대응하지 못하는 "선언적인" 결과가
    # 되기 쉽다 — 다양한 맵을 보여줘야 "장애물을 인식하고 피한다"는 일반화된
    # 행동을 배울 기회가 생긴다.
    randomize_map = args.randomize_map or (map_path is None and not args.fixed_map)
    if map_path is None and randomize_map:
        print("맵을 지정하지 않아 기본값으로 무작위 맵 학습을 사용합니다 (고정 맵을 원하면 --fixed_map).")

    if args.multiprocess:
        from rl.subproc_vec_env import SubprocVectorBipedEnv

        env_kwargs = make_env_kwargs(map_path=map_path, randomize_map=randomize_map)
        vec_env = SubprocVectorBipedEnv(
            [env_kwargs for _ in range(args.n_envs)], reset_options=reset_options
        )
        print(f"n_envs={args.n_envs} (각각 별도 프로세스, 멀티코어 병렬)")
    else:
        env_fn = make_env(map_path=map_path, randomize_map=randomize_map)
        vec_env = SyncVectorBipedEnv([env_fn for _ in range(args.n_envs)], reset_options=reset_options)
    obs_dim = vec_env.observation_space.shape[0]
    act_dim = vec_env.action_space.shape[0]
    print(f"n_envs={args.n_envs}  obs_dim={obs_dim}  act_dim={act_dim}")

    agent = PPO(
        obs_dim, act_dim,
        lr=args.lr,
        clip_ratio=args.clip_ratio,
        train_epochs=args.train_epochs,
        minibatch_size=args.minibatch_size,
    )

    obs = vec_env.reset(seed=0)

    # live_viewer를 리스트(박스)에 담아두는 이유: collect_rollout()이 --pipeline
    # 모드에서는 별도 스레드에서 실행되는데, 그 함수 안에서 "창이 닫혔다"고
    # live_viewer를 None으로 바꿔도 바깥(메인 스레드)의 일반 변수는 갱신되지
    # 않는다(파이썬 클로저는 참조가 아니라 값을 다시 바인딩하면 별개가 됨).
    # 리스트는 내용물(원소)만 바꾸는 것이므로 스레드 사이에서도 같은 객체를
    # 계속 공유해서 변경이 서로에게 보인다.
    live_viewer_box = [None]
    if args.watch:
        from rl.live_view import LiveGridViewer

        def toggle_env0_randomize(key):
            if key == ord("n"):
                e = vec_env.envs[0]
                e.randomize_map = not e.randomize_map
                obs[0] = e.reset()[0]
                live_viewer_box[0].notify_model_changed(0)
                print(f"[watch] env0 randomize_map -> {e.randomize_map}")

        live_viewer_box[0] = LiveGridViewer(
            vec_env.envs, cols=args.watch_cols, fps_cap=args.watch_fps, on_key=toggle_env0_randomize
        )
        print("실시간 시각화 창이 열렸습니다. [space]=일시정지 [+/-]=fps [n]=env0 무작위화 [q]=창 닫기")

    last_stats_box = ["아직 첫 업데이트 전..."]

    # 각 환경(N개)마다 "지금 에피소드에서 지금까지 누적한 보상"을 따로 추적.
    # 에피소드가 끝날 때(done=True) 이 값을 기록하고 0으로 리셋한다.
    ep_returns = np.zeros(args.n_envs)
    ep_lengths = np.zeros(args.n_envs, dtype=int)
    finished_returns, finished_lengths, finished_success = [], [], []

    n_iters = max(1, args.total_timesteps // (args.n_envs * args.n_steps))
    total_steps = 0
    t0 = time.time()

    def do_collect(cur_obs):
        return collect_rollout(
            vec_env, agent, cur_obs, args.n_steps, args.n_envs, obs_dim, act_dim,
            live_viewer_box, last_stats_box,
            ep_returns, ep_lengths, finished_returns, finished_lengths, finished_success,
        )

    def process_rollout(it, rollout):
        nonlocal total_steps
        buf_obs, buf_actions, buf_log_probs, buf_values, buf_rewards, buf_dones, last_values, _ = rollout
        total_steps += args.n_steps * args.n_envs

        # --- GAE 계산 ---
        # 마지막 스텝 다음의 가치도 필요하다 (에피소드가 도중에 끊겼을 때,
        # "끊긴 지점 이후로도 계속 진행됐다면 대략 이 정도 가치가 더
        # 있었을 것"이라는 추정치로 이어붙이기 위함).
        advantages, returns = compute_gae(
            buf_rewards, buf_values, buf_dones, last_values, gamma=args.gamma, lam=args.lam
        )

        # (T, N, ...) -> (T*N, ...) 로 펼쳐서 "어느 환경/시점에서 나온
        # 데이터인지" 구분 없이 하나의 큰 미니배치 풀로 섞어 학습한다.
        flat_obs = buf_obs.reshape(-1, obs_dim)
        flat_actions = buf_actions.reshape(-1, act_dim)
        flat_log_probs = buf_log_probs.reshape(-1)
        flat_advantages = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)

        # --- PPO 업데이트 ---
        stats = agent.update(flat_obs, flat_actions, flat_log_probs, flat_advantages, flat_returns)

        # --- 로그 출력 ---
        dt = time.time() - t0
        recent = slice(-50, None)
        mean_ret = np.mean(finished_returns[recent]) if finished_returns else float("nan")
        mean_len = np.mean(finished_lengths[recent]) if finished_lengths else float("nan")
        success_rate = np.mean(finished_success[recent]) if finished_success else float("nan")
        # --multiprocess일 때, 워커들이 실제로 여러 코어에 흩어져서 일하고
        # 있는지 매 이터레이션 눈으로 확인할 수 있게 CPU 사용률을 함께
        # 찍는다. (macOS는 "몇 번 코어에서 도는지" 자체를 알려주는 API가
        # 없어서, 대신 "이 프로세스가 CPU를 얼마나 쓰고 있는지"로 병렬성을
        # 보여줌 — 워커 여러 개가 동시에 80~100%씩 찍히면 실제로 병렬
        # 실행 중인 것)
        cpu_str = ""
        if hasattr(vec_env, "cpu_usage"):
            cpu = vec_env.cpu_usage()
            if cpu is not None:
                workers_str = " ".join(f"{pct:.0f}%" for _, pct in cpu["per_worker"])
                cpu_str = f"  cpu[{workers_str}]"

        print(
            f"iter {it}/{n_iters}  steps {total_steps:,}  "
            f"ep_return {mean_ret:.1f}  ep_len {mean_len:.0f}  success {success_rate:.2f}  "
            f"policy_loss {stats['policy_loss']:.4f}  value_loss {stats['value_loss']:.4f}  "
            f"entropy {stats['entropy']:.3f}  kl {stats['approx_kl']:.4f}  ({dt:.0f}s){cpu_str}"
        )
        last_stats_box[0] = [
            f"iter {it}/{n_iters}   steps {total_steps:,}   elapsed {dt:.0f}s",
            f"ep_return {mean_ret:.1f}   ep_len {mean_len:.0f}   success(goal) {success_rate*100:.0f}%",
            f"entropy {stats['entropy']:.2f}   kl {stats['approx_kl']:.4f}   value_loss {stats['value_loss']:.2f}",
        ]
        if cpu_str:
            last_stats_box[0].append(f"cpu per worker:{cpu_str}")

        if it % args.save_interval == 0 or it == n_iters:
            save_checkpoint(agent, args.ckpt_name)

    print("학습 시작. 터미널에서 Ctrl+C를 누르면 그 시점까지 학습한 내용을 저장하고 안전하게 멈춥니다.")
    if args.pipeline:
        print("[pipeline] 다음 롤아웃 수집과 지금 데이터 학습을 동시에 진행합니다.")

    try:
        if args.pipeline:
            # --- 파이프라인 모드 ---
            # 이터레이션 1의 롤아웃은 미리(동기적으로) 한 번 모아둔다. 그
            # 뒤부터는 "지금 손에 든 데이터로 학습"과 "다음 데이터를
            # 백그라운드에서 수집"을 항상 동시에 진행한다. agent.act()가
            # 학습 스레드와 무관한 ac_infer(얼린 사본)만 쓰기 때문에 두
            # 스레드가 가중치를 두고 부딪힐 일이 없다 (rl/ppo.py의
            # sync_infer 설명 참고).
            rollout = do_collect(obs)
            for it in range(1, n_iters + 1):
                obs_after = rollout[-1]
                result_box = {}

                def bg_collect(o=obs_after):
                    result_box["rollout"] = do_collect(o)

                thread = threading.Thread(target=bg_collect)
                thread.start()

                process_rollout(it, rollout)   # 메인 스레드: 지금 데이터로 학습

                thread.join()                  # 백그라운드 롤아웃이 끝날 때까지 대기
                agent.sync_infer()              # 방금 학습한 최신 가중치를 다음 롤아웃용으로 반영
                rollout = result_box["rollout"]
                obs = rollout[-1]
        else:
            for it in range(1, n_iters + 1):
                rollout = do_collect(obs)
                obs = rollout[-1]
                process_rollout(it, rollout)
    except KeyboardInterrupt:
        print("\nCtrl+C 감지 — 지금까지 학습한 모델을 저장하고 멈춥니다...")
        save_checkpoint(agent, args.ckpt_name)
        print(f"저장 완료: checkpoints/{args.ckpt_name}_actor.weights.h5 (uv run mjpython play.py 로 확인 가능)")
    else:
        print("training done.")
    finally:
        if live_viewer_box[0] is not None:
            live_viewer_box[0].close()
        vec_env.close()


if __name__ == "__main__":
    main()
