# ============================================================================
# rl/live_view.py — 학습 중인 N개 환경을 한 창에 동시에 실시간으로 보여주기
#
# MuJoCo의 인터랙티브 뷰어(mujoco.viewer)는 창 하나당 모델 하나만 보여줄 수
# 있어서, "8개 환경을 동시에" 보려면 다른 방법이 필요하다. 여기서는:
#   1) 각 환경을 화면에 안 띄우고 메모리 안에서만 렌더링(offscreen rendering)해서
#      RGB 이미지로 뽑고 (mujoco.Renderer)
#   2) N개의 이미지를 격자(grid) 모양으로 이어붙이고
#   3) 학습 통계(이터레이션, 성공률 등)를 그 위에 글자로 그려 넣은 뒤
#   4) OpenCV 창 하나로 띄운다 (cv2.imshow)
#
# 이 방식의 장점: 진짜 GPU 렌더링을 여러 창으로 띄우는 것보다 훨씬 가볍고,
# 학습 루프 안에서 완전히 제어할 수 있어 "몇 fps로 보여줄지", "일시정지",
# "새 맵으로 교체" 같은 기능을 자유롭게 넣을 수 있다.
# ============================================================================

import time

import numpy as np
import cv2
import mujoco


class LiveGridViewer:
    def __init__(self, envs, cell_size=(240, 180), cols=4, fps_cap=20,
                 window_name="training (live)", on_key=None):
        """on_key: 이 뷰어가 처리하지 않는 키(space/+/-/q 이외)를 눌렀을 때
        호출되는 콜백 on_key(key: int). train.py는 이걸로 'n' 키를 받아서
        "환경 하나를 무작위 맵으로 바꿔달라" 같은 동적 기능을 붙인다.
        """
        self.envs = envs
        self.cell_w, self.cell_h = cell_size
        self.cols = cols
        self.rows = (len(envs) + cols - 1) // cols
        self.fps_cap = fps_cap
        self.window_name = window_name
        self.on_key = on_key
        self._last_render_t = 0.0
        # 환경마다 독립된 offscreen 렌더러 하나씩 (렌더러는 자기 모델의
        # 크기/구조에 맞춰 GPU 버퍼를 미리 준비해두므로 매번 새로 만들지 않고 재사용)
        self._renderers = [mujoco.Renderer(e.model, height=self.cell_h, width=self.cell_w) for e in envs]
        self.paused = False
        self.quit_requested = False

    def _rebuild_renderer(self, idx):
        """해당 환경의 모델이 바뀌었을 때(맵 교체 등) 렌더러도 새로 맞춰준다."""
        self._renderers[idx].close()
        self._renderers[idx] = mujoco.Renderer(
            self.envs[idx].model, height=self.cell_h, width=self.cell_w
        )

    def notify_model_changed(self, idx):
        self._rebuild_renderer(idx)

    def maybe_update(self, stats_lines):
        """호출은 매 스텝 해도 되지만, 내부적으로 fps_cap을 넘지 않도록
        스스로 프레임을 건너뛴다 (렌더링이 무거우므로, 학습 속도를 지키기
        위해 너무 자주 실제로 그리지는 않음). 반환값 False면 종료 요청됨.
        """
        now = time.time()
        if self.paused:
            key = cv2.waitKey(30) & 0xFF
            self._handle_key(key)
            return not self.quit_requested

        if now - self._last_render_t < 1.0 / self.fps_cap:
            return not self.quit_requested
        self._last_render_t = now

        tiles = []
        for env, renderer in zip(self.envs, self._renderers):
            renderer.update_scene(env.data)
            frame = renderer.render()  # RGB, (H, W, 3), uint8
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            tiles.append(frame_bgr)

        grid = self._compose_grid(tiles)
        self._draw_overlay(grid, stats_lines)

        cv2.imshow(self.window_name, grid)
        key = cv2.waitKey(1) & 0xFF
        self._handle_key(key)
        return not self.quit_requested

    def _handle_key(self, key):
        if key == 255:  # 아무 키도 안 눌림 (cv2.waitKey의 기본 반환값)
            return
        if key == ord("q"):
            self.quit_requested = True
        elif key == ord(" "):
            self.paused = not self.paused
        elif key == ord("+") or key == ord("="):
            self.fps_cap = min(60, self.fps_cap + 5)
        elif key == ord("-"):
            self.fps_cap = max(1, self.fps_cap - 5)
        elif self.on_key is not None:
            self.on_key(key)

    def _compose_grid(self, tiles):
        h, w = self.cell_h, self.cell_w
        grid = np.zeros((self.rows * h, self.cols * w, 3), dtype=np.uint8)
        for i, tile in enumerate(tiles):
            r, c = divmod(i, self.cols)
            grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = tile
        return grid

    def _draw_overlay(self, grid, stats_lines):
        # 화면 위쪽에 반투명 검은 띠를 깔고 그 위에 통계 텍스트를 그린다
        band_h = 20 * (len(stats_lines) + 1)
        overlay = grid.copy()
        cv2.rectangle(overlay, (0, 0), (grid.shape[1], band_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, grid, 0.45, 0, dst=grid)
        for i, line in enumerate(stats_lines):
            cv2.putText(
                grid, line, (8, 18 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )
        cv2.putText(
            grid, "[space]=pause  [+/-]=fps  [n]=env0 맵 무작위화 토글  [q]=창 닫기(학습 계속)",
            (8, grid.shape[0] - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 220, 255), 1, cv2.LINE_AA,
        )

    def close(self):
        for r in self._renderers:
            r.close()
        cv2.destroyWindow(self.window_name)
