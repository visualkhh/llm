# ============================================================================
# nodes/core.py — ROS2를 흉내낸 아주 작은 노드 프레임워크
#
# 진짜 ROS2는 각 노드가 별도 프로세스일 수도 있고, DDS라는 네트워크
# 프로토콜로 토픽을 주고받는다. 이 시뮬레이터는 전부 한 프로세스 안에서
# 돌아가므로 그렇게까지 무거울 필요가 없다 — 대신 "구조"만 똑같이
# 흉내낸다:
#
#   - Node: 이름을 갖고, publisher/subscriber를 만들고, 자기만의 주기
#     (rate_hz)로 실행되는 타이머 콜백을 가질 수 있다.
#   - Bus: 토픽 이름 -> 구독자 콜백 목록을 들고 있는 아주 단순한
#     "메시지 브로커". publish()하면 그 토픽을 구독 중인 모든 콜백이
#     즉시(동기적으로, 같은 함수 호출 스택 안에서) 실행된다.
#   - Executor: 등록된 모든 노드의 타이머를 각자의 주기에 맞춰 돌려주는
#     스케줄러. 라이다는 10Hz, IMU는 100Hz처럼 센서마다 원하는 주기가
#     다르므로, "다음 실행 시각"을 각 타이머마다 따로 관리한다.
#
# 진짜 ROS2와 다른 점(의도적 단순화): 메시지는 msgs.py의 dataclass
# 인스턴스를 그대로 넘긴다(직렬화 없음, 어차피 같은 프로세스라 복사할
# 필요가 없다) — 하지만 "이 토픽엔 이 타입만 흐른다"는 진짜 ROS2의
# 타입 계약은 Bus가 그대로 강제한다 (아래 _declare_type/publish 참고).
# ============================================================================

import time
from dataclasses import dataclass, field


class Bus:
    """토픽 기반 pub/sub. 여러 노드가 공유하는 단 하나의 인스턴스를 쓴다.

    진짜 ROS2에서는 토픽 하나에 항상 딱 하나의 메시지 타입만 흐른다 —
    `ros2 topic info /cmd_vel`을 치면 "Type: geometry_msgs/msg/Twist"라고
    나오고, 다른 타입의 퍼블리셔/서브스크라이버를 만들면 아예 안 만들어
    지거나 경고가 뜬다. 여기서도 그 계약을 흉내낸다: 어떤 토픽에 처음
    선언된 메시지 타입을 기억해두고, 그 뒤로 다른 타입을 발행/구독하려
    하면 바로 에러를 낸다 — 그래야 "엉뚱한 타입을 잘못 보냈는데 아무도
    눈치 못 채는" 버그를 코드 작성 시점에 바로 잡을 수 있다.
    """

    def __init__(self):
        self._subscribers = {}   # topic -> list[callback]
        self._latest = {}        # topic -> 가장 최근 발행된 메시지 (뒤늦게 구독해도 최신값 조회 가능)
        self._topic_types = {}   # topic -> 이 토픽에 흘러야 하는 메시지 타입(class)

    def declare_type(self, topic, msg_type):
        existing = self._topic_types.get(topic)
        if existing is not None and existing is not msg_type:
            raise TypeError(
                f"토픽 '{topic}'은 이미 {existing.__name__} 타입으로 선언되어 있는데 "
                f"{msg_type.__name__}로 다시 선언하려 했습니다 (create_publisher/"
                f"create_subscription에 msg_type을 다르게 준 곳이 있는지 확인하세요)."
            )
        self._topic_types[topic] = msg_type

    def publish(self, topic, message):
        expected = self._topic_types.get(topic)
        if expected is not None and not isinstance(message, expected):
            raise TypeError(
                f"토픽 '{topic}'에는 {expected.__name__} 타입만 흘러야 하는데 "
                f"{type(message).__name__} 인스턴스를 발행하려 했습니다."
            )
        self._latest[topic] = message
        for callback in self._subscribers.get(topic, []):
            callback(message)

    def subscribe(self, topic, callback, msg_type=None):
        if msg_type is not None:
            self.declare_type(topic, msg_type)
        self._subscribers.setdefault(topic, []).append(callback)

    def latest(self, topic, default=None):
        return self._latest.get(topic, default)


@dataclass
class _Timer:
    node_name: str
    rate_hz: float
    callback: callable
    next_run: float = field(default=0.0)

    @property
    def period(self):
        return 1.0 / self.rate_hz


class Node:
    """모든 노드(바퀴/조인트/센서/로컬라이제이션/내비게이션/매니퓰레이션)의
    공통 부모. rclpy.Node의 아주 작은 흉내."""

    def __init__(self, name, bus: Bus):
        self.name = name
        self.bus = bus
        self._timers = []

    def create_publisher(self, topic, msg_type=None):
        """토픽에 발행하는 함수를 하나 돌려준다 (rclpy의
        create_publisher(msg_type, topic, qos).publish(msg)와 같은 용도 —
        여기서는 QoS는 생략하고 msg_type만 선택적으로 받는다. msg_type을
        주면 이 토픽에 그 타입 말고 다른 걸 보내려 할 때 Bus가 에러를 낸다)."""
        if msg_type is not None:
            self.bus.declare_type(topic, msg_type)
        return lambda message: self.bus.publish(topic, message)

    def create_subscription(self, topic, callback, msg_type=None):
        self.bus.subscribe(topic, callback, msg_type=msg_type)

    def create_timer(self, rate_hz, callback):
        """이 노드가 rate_hz(Hz)마다 callback()을 실행하고 싶다고 등록.
        실제 실행은 Executor가 담당한다 (여러 노드의 타이머를 한 곳에서
        스케줄링해야 "각 센서가 자기 주기대로" 도는 걸 관리하기 쉽다)."""
        timer = _Timer(node_name=self.name, rate_hz=rate_hz, callback=callback)
        self._timers.append(timer)
        return timer


class ComposableNode(Node):
    """ROS2의 "Composable Node"에 해당하는 개념 — odometry/SLAM처럼 여러
    센서 데이터를 조합(compose)해서 더 고차원의 결과(위치 추정 등)를
    만들어내는 노드. 일반 Node와 실행 방식은 완전히 같지만(이 시뮬레이터는
    어차피 전부 한 프로세스 안에 있어서 "프로세스 내 결합"이라는 원래
    의미가 자동으로 성립한다), "센서 원값을 만드는 노드"와 "여러 값을
    엮어 해석하는 노드"를 코드 구조상 구분하기 위해 별도 이름을 준다.
    """
    pass


class Executor:
    """등록된 노드들의 모든 타이머를 모아서, 각자의 주기에 맞춰 실행한다."""

    def __init__(self):
        self.nodes = []
        self._timers = []

    def add_node(self, node: Node):
        self.nodes.append(node)
        self._timers.extend(node._timers)

    def spin_once(self, now=None):
        """지금 시각 기준으로, 실행할 때가 된 타이머들을 전부 한 번씩 실행."""
        now = time.time() if now is None else now
        for timer in self._timers:
            if now >= timer.next_run:
                timer.callback()
                timer.next_run = now + timer.period

    def spin_for(self, duration_sec, tick_hz=200.0):
        """duration_sec 동안 spin_once를 tick_hz 주기로 반복 (테스트/헤드리스
        실행용 — GUI 루프에서는 매 프레임 spin_once만 직접 호출한다)."""
        start = time.time()
        tick = 1.0 / tick_hz
        while time.time() - start < duration_sec:
            self.spin_once()
            time.sleep(tick)
