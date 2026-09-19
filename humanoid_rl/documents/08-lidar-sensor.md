# 08. 라이다(거리 센서) — 카메라 대신 레이캐스팅을 쓴 이유

관련 코드: `envs/biped_env.py`의 `_lidar_scan`, `maps/generate_map.py`의 `SENSED_GEOM_GROUP`

## 왜 필요했나 — "치팅"이었던 이전 방식

이전 버전은 장애물의 정확한 world 좌표를 로봇 기준 상대좌표로 바꿔서
관측값에 그대로 넣어줬습니다. 문제는 이게 실제 로봇이라면 있을 수 없는
정보라는 것 — 진짜 로봇은 자기 몸에 달린 센서(카메라, 라이다, 초음파
등)로 "감지한 것"만 알 수 있지, 장애물이 정확히 몇 미터 몇 도 방향에
있는지를 마법처럼 알 수는 없습니다. 이런 식으로 환경이 정답에 가까운
정보를 미리 계산해 그냥 던져주는 것을, 실제로는 얻을 수 없는 정보라는
뜻에서 **특권 정보(privileged information)** 라고 부릅니다.

## 카메라 대신 라이다를 고른 이유

가장 "리얼"한 방법은 로봇 머리에 카메라를 달아 실제 이미지를 렌더링해서
그걸 CNN(합성곱 신경망)으로 처리하는 것입니다. 하지만:

- 매 스텝 이미지를 렌더링하는 비용이 물리 연산보다 훨씬 커서, 8~16개
  환경을 동시에 돌리는 이 프로젝트 구조에서 학습이 수십 배 느려질 수 있음
- 픽셀에서 직접 배우는 강화학습(pixel-based RL)은 원래 샘플 효율이
  나쁘기로 유명해서, 훨씬 더 오래/많이 학습시켜야 함
- CNN 특징 추출기가 필요해 모델이 커지고 구현도 복잡해짐

그래서 실제 다리로봇 연구에서도 흔히 쓰이는 절충안인 **라이다(레이저
거리 센서) 스타일**을 택했습니다. 부채꼴로 광선을 여러 개 쏴서 "이
방향으로 몇 미터 앞에 뭔가 있다"는 거리값만 얻는 방식으로, 이미지
렌더링 없이 MuJoCo의 레이캐스팅 함수(`mj_ray`) 하나로 매우 저렴하게
계산할 수 있으면서도, 장애물 좌표를 직접 주지 않는다는 핵심 원칙은
그대로 지킵니다.

## 어떻게 동작하는가

```python
def _lidar_scan(self, yaw):
    origin = self.data.xpos[self._head_id].copy()
    half_fov = np.radians(LIDAR_FOV_DEG) / 2.0
    angles = yaw + np.linspace(-half_fov, half_fov, N_LIDAR_RAYS)
    ...
    for angle in angles:
        vec = [cos(angle), sin(angle), 0]
        d = mujoco.mj_ray(model, data, origin, vec, geomgroup, ...)
        dists.append(LIDAR_MAX_RANGE if d < 0 else min(d, LIDAR_MAX_RANGE))
```

- **발사 지점**: 로봇의 머리(`head`) 위치. 실제로 센서를 머리에 단
  것처럼 로봇이 이동/회전하면 광선 발사 지점도 함께 움직인다.
- **부채꼴 각도**: 로봇이 지금 바라보는 방향(`yaw`)을 기준으로 좌우
  ±60도(총 120도, `LIDAR_FOV_DEG`). 사람의 대략적인 좌우 시야각과
  비슷한 정도로 잡았다.
- **광선 개수**: 15개(`N_LIDAR_RAYS`). 이 부채꼴 안에 균등한 간격으로
  15개를 쏘아 해상도를 정한다. 늘리면 더 정밀하게 주변을 파악할 수
  있지만 관측값 차원이 커져 학습이 조금 더 어려워질 수 있다.
- **최대 거리**: 6m(`LIDAR_MAX_RANGE`). 이보다 멀리 있거나 아무것도
  안 걸리면(레이가 허공으로 나가면 `mj_ray`가 `-1`을 반환) "이 방향은
  비어있다"는 뜻으로 최댓값을 채운다 — 관측값 크기를 항상 일정하게
  유지하기 위함이다 (몇 개가 감지됐는지와 무관하게 항상 15개 숫자).

## 로봇 자기 몸은 왜 안 걸리는가 — geom group 필터링

레이가 별도 처리 없이 아무 geom이나 다 감지하게 두면, 로봇 자신의
팔다리나 몸통에 레이가 바로 부딪혀서 "0.01m 앞에 뭔가 있다"는 식의
무의미한 값만 나오게 됩니다. 이를 막기 위해 MuJoCo의 **geom group**
기능을 사용합니다 — 모든 geom은 0~5번 그룹 중 하나에 속할 수 있고,
레이캐스팅 시 "이 그룹들만 본다"고 필터링할 수 있습니다.

```python
# maps/generate_map.py
SENSED_GEOM_GROUP = 1   # 지면/장애물은 이 그룹
...
spec.worldbody.add_geom(..., group=SENSED_GEOM_GROUP)  # ground, obstacle_N에 적용
```

```python
# envs/biped_env.py
_LIDAR_GEOMGROUP = np.zeros(6, dtype=np.uint8)
_LIDAR_GEOMGROUP[SENSED_GEOM_GROUP] = 1   # group 1만 감지하도록 필터
```

로봇(URDF에서 온 geom들)은 별도로 그룹을 지정하지 않아 기본값인
group 0에 남아있고, 라이다는 group 1만 보도록 필터링되어 있어서
로봇 자기 몸은 자동으로 무시됩니다. 직접 실험으로 검증한 내용:

```python
# 로봇을 향해 레이를 쏴도(자기 몸 관통 방향) group 1만 필터링하면 감지 안 됨
dist = mj_ray(model, data, head_pos, toward_own_body, geomgroup=[0,1,0,0,0,0], ...)
# -> -1.0 (검출 없음, 정상)

# 실제 장애물을 향해 쏘면 정확히 감지됨
dist = mj_ray(model, data, head_pos, toward_obstacle, geomgroup=[0,1,0,0,0,0], ...)
# -> 1.6 (obstacle_0까지의 실제 거리)
```

## 사용자가 직접 맵을 만들 때 주의할 점

`maps/generate_map.py`로 생성하면 자동으로 `group="1"`이 붙지만,
**맵 파일을 손으로 직접 작성/수정할 경우** 장애물 geom에 `group="1"`을
꼭 넣어야 라이다가 그 장애물을 감지할 수 있습니다. 빠뜨리면 물리적으로는
충돌하지만(로봇이 못 지나감) 센서에는 "안 보이는" 장애물이 되어, 정책이
미리 피할 방법 없이 갑자기 부딪히게 됩니다 ([06-map-files.md](06-map-files.md)
스키마 설명 참고).

## play.py에서 확인해보기

`--print_ctrl`이 관절 컨트롤 값을 보여주듯, 라이다 값 자체를 직접 눈으로
확인하고 싶다면 `env._lidar_scan(yaw)`를 호출해보거나, 관측값의 마지막
15개(`obs[-15:]`)를 출력해보면 됩니다 — 정면에 장애물이 가까워질수록
해당 방향의 숫자가 작아지는 것을 볼 수 있습니다.
