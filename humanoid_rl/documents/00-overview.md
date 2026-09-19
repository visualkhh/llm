# 00. 전체 개요

이 프로젝트는 **2족보행 휴머노이드 로봇**을 MuJoCo 물리 시뮬레이터 안에서
직접 만들고(URDF), 장애물이 있는 환경에서 **목표 지점으로 이동하면 점수를
얻는 강화학습(PPO)**으로 스스로 걷는 법을 배우게 만드는 프로젝트입니다.

## 전체 파이프라인

```
robot/biped.urdf   maps/map_default.xml (또는 무작위 생성 맵)
로봇 뼈대만 정의    지면 + 장애물만 정의 (URDF와 완전히 독립된 파일)
        │                   │
        └─────────┬─────────┘
                   ▼  envs/biped_env.py — MjSpec.attach()로 조립
      [로봇] + [맵] + [목표 마커] + [10개 다리 모터]
                   │
                   ▼  Gymnasium 표준 인터페이스로 감싸짐 (reset/step)
              BipedEnv  (하나의 완결된 강화학습 환경)
        │
        ▼  rl/vec_env.py — 여러 개를 동시에 돌림 (N=8)
     SyncVectorBipedEnv
        │
        ▼  rl/ppo.py — Actor-Critic 신경망 + PPO 알고리즘
     train.py 가 반복 학습 → checkpoints/*.weights.h5 저장
        │
        ▼
     play.py — 학습된 정책을 MuJoCo 뷰어로 실시간 시각화
```

## 파일 구조

| 경로 | 역할 | 관련 문서 |
|---|---|---|
| `robot/biped.urdf` | 로봇의 물리적 정의 (관절, 질량, 크기) | [01-urdf-and-mujoco.md](01-urdf-and-mujoco.md) |
| `maps/*.xml`, `maps/generate_map.py` | 지면/장애물 배치 — 로봇과 분리된 독립 파일 | [06-map-files.md](06-map-files.md) |
| `envs/biped_env.py` | 로봇+맵 조립 + 관측값/보상 설계 | [02-environment-and-reward.md](02-environment-and-reward.md) |
| `rl/ppo.py`, `rl/vec_env.py` | PPO 강화학습 알고리즘 | [03-ppo-algorithm.md](03-ppo-algorithm.md) |
| `train.py` | 학습 루프 실행 | [04-training-loop.md](04-training-loop.md) |
| `play.py` | 학습 결과를 눈으로 확인 | [04-training-loop.md](04-training-loop.md) |
| `select_scenario.py`, `rl/live_view.py` | 맵/목표 선택 + 실시간 다중 환경 시각화 | [07-scenario-and-live-view.md](07-scenario-and-live-view.md) |

## 이번 1단계 목표

> 장애물이 있는 평지에서, 매 에피소드마다 (또는 사용자가 지정한) 목표
> 지점을 향해 로봇이 걸어가면 점수를 받는다. 목표에 가까워질수록,
> 넘어지지 않을수록 점수가 오른다.

로봇은 **이미 두 발로 서 있는 자세에서 시작**합니다. "땅에 쓰러진 채로
시작해서 스스로 일어나는 법"까지 배우는 것은 훨씬 더 어려운 다음 단계
문제이므로, 1단계에서는 의도적으로 범위를 좁혔습니다.

## 현실적으로 기대할 수 있는 것

사람이 아무 사전 지식 없이 두 다리로 걷는 법을 강화학습만으로(모방학습
없이, 순수 시행착오로) 익히는 것은 로보틱스/강화학습 분야에서도 잘 알려진
**어려운 문제**입니다. 유명한 벤치마크(OpenAI Gym의 Humanoid-v4 등)에서도
그럴듯한 걸음걸이가 나오기까지 보통 수백만~수천만 스텝, 여러 개의 병렬
환경, 상당한 보상 함수 튜닝이 필요합니다.

이 프로젝트는:
- CPU 한 대에서 몇 시간 안에 끝낼 수 있는 규모로 축소되어 있고,
- 처음 몇백만 스텝만으로는 "우아하게 걷기"보다는 "넘어지지 않으려
  버둥대며 조금씩 전진"하는 수준에 그칠 가능성이 높습니다.

이건 버그가 아니라 **강화학습으로 보행을 배우는 문제의 원래 난이도**입니다.
학습이 잘 안 되는 것처럼 보일 때 무엇을 점검해야 하는지는
[05-next-steps.md](05-next-steps.md)에 정리했습니다.
