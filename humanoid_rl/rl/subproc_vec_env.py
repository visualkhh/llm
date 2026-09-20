# ============================================================================
# rl/subproc_vec_env.py — 환경 N개를 "진짜" 여러 CPU 코어에서 병렬로 돌리기
#
# rl/vec_env.py의 SyncVectorBipedEnv는 파이썬 for문으로 환경을 하나씩
# 차례로 진행시킨다 — 코드는 단순하지만, 결국 이 한 프로세스 안에서
# 순서대로 실행되므로 CPU 코어 하나만 쓰고 나머지 코어는 놀게 된다
# (파이썬의 GIL 때문에, 같은 프로세스 안에서는 스레드를 여러 개 써도
# 물리 시뮬레이션 같은 CPU 연산은 진짜 동시에 실행되지 않는다).
#
# 이 파일은 그 대신 환경 하나마다 완전히 별도의 OS 프로세스를 하나씩
# 띄운다. 프로세스는 스레드와 달리 GIL의 제약을 받지 않으므로, 운영체제가
# 각 프로세스를 서로 다른 CPU 코어에 진짜로 동시에 배정해 실행한다.
# 메인 프로세스와 각 워커(worker) 프로세스는 pipe(파이프)로 명령/결과를
# 주고받는다.
#
#   메인 프로세스                    워커 프로세스 0        워커 프로세스 1   ...
#   ("step", action0) ────────────▶  BipedEnv.step()
#   ("step", action1) ─────────────────────────────────▶  BipedEnv.step()
#   (모든 워커에 명령을 다 보낸 뒤)
#   result0 ◀────────────────────── (obs, reward, ...)
#   result1 ◀───────────────────────────────────────────  (obs, reward, ...)
#
# 핵심은 "명령을 모든 워커에 먼저 다 보내고, 그 다음에 결과를 모아서
# 받는다"는 순서다. 이렇게 해야 N개의 물리 시뮬레이션이 실제로 동시에
# (병렬로) 진행되고, 메인 프로세스는 그 결과들이 도착하기를 기다리기만
# 하면 된다.
#
# 주의: 이 방식은 각 환경이 별도 프로세스에 있으므로, train.py의
# --watch(실시간 시각화)처럼 "각 환경의 model/data 객체를 메인 프로세스에서
# 직접 들여다보는" 기능과는 함께 쓸 수 없다. 시각화가 필요 없는 순수
# 학습 속도가 중요할 때 사용한다.
# ============================================================================

import multiprocessing as mp
import signal
import threading

import numpy as np

try:
    import psutil
except ImportError:
    psutil = None

from envs.biped_env import BipedEnv


def _worker(remote, env_kwargs):
    """자식 프로세스에서 실행되는 함수. 파이프로 명령을 받아 로컬 환경에 적용하고
    결과를 다시 파이프로 돌려보내는 무한 루프.

    반드시 모듈 최상위 함수여야 한다 — macOS/Windows의 멀티프로세싱 기본
    방식(spawn)은 자식 프로세스를 만들 때 이 함수를 다시 import해서 찾기
    때문에, 클래스 안의 메서드나 클로저로는 안 된다.

    Ctrl+C(SIGINT)는 터미널이 "포그라운드 프로세스 그룹 전체"에 보내기
    때문에, 이 자식 프로세스도 메인 프로세스와 동시에 SIGINT를 직접
    받는다. 여기서 그걸 그대로 처리해버리면(=KeyboardInterrupt 발생) 메인
    프로세스가 아직 "step" 결과를 기다리는 도중에 이 워커가 응답 없이
    죽어버려서, 메인 쪽 파이프가 EOFError로 끊기고 뒤이어
    vec_env.close()의 종료 명령 전송도 BrokenPipeError로 실패하게 된다.
    그래서 워커는 SIGINT를 아예 무시하고, 메인 프로세스가 명시적으로
    보내는 ("close", None) 명령을 받았을 때만 정상 종료한다.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    env = BipedEnv(**env_kwargs)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                obs, reward, terminated, truncated, info = env.step(data)
                done = terminated or truncated
                if done:
                    # SyncVectorBipedEnv와 동일한 동작: 에피소드가 끝나면
                    # 그 워커만 즉시 다음 에피소드를 이어서 시작한다.
                    info = dict(info)
                    info["terminal_observation"] = obs
                    obs, _ = env.reset()
                remote.send((obs, reward, done, info))
            elif cmd == "reset":
                seed, options = data
                obs, _ = env.reset(seed=seed, options=options)
                remote.send(obs)
            elif cmd == "close":
                env.close()
                remote.close()
                break
            else:
                raise ValueError(f"unknown command: {cmd}")
    except (EOFError, KeyboardInterrupt):
        env.close()


class SubprocVectorBipedEnv:
    """SyncVectorBipedEnv와 동일한 인터페이스(reset/step/close)를 갖지만,
    내부적으로 환경 하나당 프로세스 하나를 띄워 진짜 병렬로 실행한다.
    """

    def __init__(self, env_kwargs_list, reset_options=None):
        self.n = len(env_kwargs_list)
        self.reset_options = reset_options

        # 관측/행동 공간 크기를 알아내기 위해 임시로 환경 하나를 메인
        # 프로세스에서 만들어봄 (워커들과는 무관한, 크기 확인용 인스턴스).
        probe_env = BipedEnv(**env_kwargs_list[0])
        self.observation_space = probe_env.observation_space
        self.action_space = probe_env.action_space
        probe_env.close()

        # --pipeline 모드에서는 메인 스레드(close())와 백그라운드 스레드
        # (step())가 동시에 이 파이프들을 건드릴 수 있다. Ctrl+C를 급하게
        # 여러 번 눌러 정리 단계의 bg_thread.join()마저 중간에 끊기면, 두
        # 스레드가 같은 파이프에 동시에 send()를 시도해서 "close" 명령을
        # 받은 워커가 파이프를 닫는 순간 다른 스레드의 send()가
        # BrokenPipeError로 죽는 경쟁 상태(race condition)가 생길 수
        # 있다. step()/close()가 서로 겹쳐 실행되지 못하게 락으로 막는다.
        self._lock = threading.Lock()

        ctx = mp.get_context("spawn")
        self.remotes, worker_remotes = zip(*[ctx.Pipe() for _ in range(self.n)])
        self.processes = [
            ctx.Process(target=_worker, args=(worker_remote, kwargs), daemon=True)
            for worker_remote, kwargs in zip(worker_remotes, env_kwargs_list)
        ]
        for p in self.processes:
            p.start()

        # 어떤 워커가 실제로 어떤 OS 프로세스(PID)로 떴는지 확인할 수 있게
        # 출력해준다 — 활동 모니터(macOS)나 `ps`/`top`에서 이 PID들을 찾아보면
        # 서로 다른 프로세스로 진짜 떠 있는 걸 직접 확인할 수 있다.
        pids = [p.pid for p in self.processes]
        print(f"[multiprocess] {self.n}개 워커 프로세스 시작됨. PID: {pids}")

        # psutil로 각 워커 프로세스의 CPU 사용률을 나중에 조회할 수 있게
        # Process 핸들을 미리 만들어둔다. macOS는 "이 프로세스가 지금 몇 번
        # 코어에서 도는지"를 알려주는 API(cpu_num 등)를 지원하지 않아서
        # (리눅스/윈도우 전용), 대신 "각 워커가 실제로 CPU를 얼마나 쓰고
        # 있는지(cpu_percent)"로 병렬 실행이 되고 있는지를 확인한다.
        self._psutil_procs = [psutil.Process(pid) for pid in pids] if psutil else None
        if self._psutil_procs:
            for proc in self._psutil_procs:
                proc.cpu_percent()  # 첫 호출은 기준점을 잡기 위한 것이라 값 자체는 버림
            psutil.cpu_percent(percpu=True)  # 시스템 전체 코어별 측정도 마찬가지로 기준점 잡기

    def cpu_usage(self):
        """각 워커 프로세스의 (PID, 최근 CPU 사용률%) 목록과, 시스템 전체
        코어별 사용률을 반환한다. psutil이 없으면 None.
        """
        if not self._psutil_procs:
            return None
        per_worker = [(p.pid, p.cpu_percent()) for p in self._psutil_procs]
        per_core = psutil.cpu_percent(percpu=True)
        return {"per_worker": per_worker, "per_core": per_core}

    def reset(self, seed=None):
        with self._lock:
            for i, remote in enumerate(self.remotes):
                s = None if seed is None else seed + i
                remote.send(("reset", (s, self.reset_options)))
            obs_list = [remote.recv() for remote in self.remotes]
        return np.stack(obs_list)

    def step(self, actions):
        with self._lock:
            # 1) 모든 워커에게 먼저 명령을 다 보낸다 (여기서 각 워커가 동시에
            #    자기 물리 시뮬레이션을 진행하기 시작함)
            for remote, action in zip(self.remotes, actions):
                remote.send(("step", action))
            # 2) 그 다음에 결과를 순서대로 모은다 (이미 다 병렬로 계산되고 있으므로
            #    여기서 기다리는 시간은 "가장 느린 워커 하나"의 시간과 비슷하다 —
            #    N개를 순서대로 계산할 때보다 훨씬 빠르다)
            results = [remote.recv() for remote in self.remotes]
        obs, rewards, dones, infos = zip(*results)
        return np.stack(obs), np.array(rewards, dtype=np.float32), np.array(dones, dtype=bool), list(infos)

    def close(self):
        # 워커는 이제 SIGINT를 무시하므로 정상적인 상황에서는 항상 살아
        # 있지만, 혹시라도 이미 죽어 있는 워커가 있어도(예: 물리 시뮬레이션
        # 내부 에러로 먼저 죽은 경우) close() 자체가 통째로 실패해서 나머지
        # 정리 작업(체크포인트 저장 등)까지 막지 않도록 방어적으로 처리한다.
        # step()/reset()과 같은 락을 쓰므로, 백그라운드 스레드가 마침
        # step() 도중이면 그게 끝날 때까지 여기서 기다렸다가 안전하게
        # close 명령을 보낸다 (파이프 동시 접근으로 인한 BrokenPipeError 방지).
        with self._lock:
            for remote in self.remotes:
                try:
                    remote.send(("close", None))
                except (BrokenPipeError, OSError):
                    pass
        for p in self.processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
