import math
import threading
import time
from dataclasses import dataclass, asdict
from typing import List, Optional

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from pathlib import Path
from ament_index_python.packages import get_package_share_directory

# =========================
# Utils
# =========================
def yaw_to_quat(yaw: float):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return (0.0, 0.0, sy, cy)  # x,y,z,w


# =========================
# Data Models
# =========================
@dataclass
class Task:
    task_id: int
    x: float
    y: float
    yaw: float
    status: str = "PENDING"  # PENDING | RUNNING | DONE | FAILED
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None


class CreateTaskReq(BaseModel):
    x: float
    y: float
    yaw: float = 0.0
    # 필요하면 확장: max_retries 등


# =========================
# FMS Core Node (Single Robot)
# - 실패해도 IDLE 복구하고 다음 작업 계속 수행
# =========================
class FMSServer(Node):
    def __init__(self):
        super().__init__("fms_server")

        # use_sim_time 은 CLI에서 -p use_sim_time:=True 로 주입 (여기서 declare 안 함)
        self.declare_parameter("frame_id", "map")
        self.frame_id = self.get_parameter("frame_id").value

        self.nav_client = ActionClient(self, NavigateToPose, "navigate_to_pose")

        self._lock = threading.Lock()
        self._task_id_seq = 1
        self._tasks: List[Task] = []

        self.robot_status = "IDLE"  # IDLE | MOVING
        self.current_task_id: Optional[int] = None
        self.last_error: Optional[str] = None

        self._stop = False
        self._worker = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._worker.start()

        self.get_logger().info("FMS started (single robot)")

    def shutdown(self):
        self._stop = True
        try:
            self._worker.join(timeout=2.0)
        except Exception:
            pass

    # ----- API helpers -----
    def add_task(self, x: float, y: float, yaw: float) -> Task:
        with self._lock:
            t = Task(task_id=self._task_id_seq, x=float(x), y=float(y), yaw=float(yaw))
            self._task_id_seq += 1
            self._tasks.append(t)
            return t

    def get_state(self):
        with self._lock:
            return {
                "robot": {
                    "status": self.robot_status,
                    "current_task_id": self.current_task_id,
                    "last_error": self.last_error,
                },
                "tasks": [asdict(t) for t in self._tasks],
            }

    # ----- scheduler loop -----
    def _scheduler_loop(self):
        while rclpy.ok() and not self._stop:
            if self.nav_client.wait_for_server(timeout_sec=1.0):
                break
            self.get_logger().info("Waiting for Nav2 action server...")

        while rclpy.ok() and not self._stop:
            task = None
            with self._lock:
                if self.robot_status == "IDLE":
                    for t in self._tasks:
                        if t.status == "PENDING":
                            task = t
                            t.status = "RUNNING"
                            t.started_at = time.time()
                            t.finished_at = None
                            t.error = None
                            self.robot_status = "MOVING"
                            self.current_task_id = t.task_id
                            self.last_error = None
                            break

            if task is None:
                time.sleep(0.05)
                continue

            ok, err = self._navigate_to(task.x, task.y, task.yaw)

            with self._lock:
                # ✅ 성공/실패 관계없이 다음 작업을 위해 IDLE 복구
                self.robot_status = "IDLE"
                self.current_task_id = None

                if ok:
                    task.status = "DONE"
                    task.finished_at = time.time()
                    task.error = None
                    self.last_error = None
                else:
                    task.status = "FAILED"
                    task.finished_at = time.time()
                    task.error = err or "unknown"
                    self.last_error = task.error

            # 실패 직후 살짝 텀(옵션)
            if not ok:
                time.sleep(0.2)

    # ----- Nav2 Action (callback + event) -----
    def _navigate_to(self, x: float, y: float, yaw: float):
        goal = NavigateToPose.Goal()
        pose = PoseStamped()
        pose.header.frame_id = self.frame_id
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)

        qx, qy, qz, qw = yaw_to_quat(float(yaw))
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        goal.pose = pose

        done_evt = threading.Event()
        outcome = {"ok": False, "err": "unknown"}

        def _on_result(fut):
            try:
                res = fut.result()
                if res is None:
                    outcome["ok"] = False
                    outcome["err"] = "no_result"
                    return
                status = int(res.status)
                # 일반적으로 4=SUCCEEDED
                if status == 4:
                    outcome["ok"] = True
                    outcome["err"] = None
                else:
                    outcome["ok"] = False
                    outcome["err"] = f"nav2_failed_status={status}"
            except Exception as e:
                outcome["ok"] = False
                outcome["err"] = f"result_exception: {e}"
            finally:
                done_evt.set()

        def _on_goal_sent(fut):
            try:
                gh = fut.result()
                if (gh is None) or (not gh.accepted):
                    outcome["ok"] = False
                    outcome["err"] = "goal_rejected"
                    done_evt.set()
                    return
                gh.get_result_async().add_done_callback(_on_result)
            except Exception as e:
                outcome["ok"] = False
                outcome["err"] = f"goal_send_exception: {e}"
                done_evt.set()

        self.nav_client.send_goal_async(goal).add_done_callback(_on_goal_sent)

        if not done_evt.wait(timeout=180.0):
            return False, "timeout_waiting_result"

        return outcome["ok"], outcome["err"]


# =========================
# FastAPI + Web UI (rosbridge + roslibjs 방식)
# =========================
app = FastAPI()
_ros_node: Optional[FMSServer] = None

# static/roslib.min.js 제공
STATIC_DIR = Path(get_package_share_directory("fms_server")) / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse("FMS_web-interface/src/fms_server/fms_server/templates/index.html")


@app.post("/tasks")
def create_task(req: CreateTaskReq):
    assert _ros_node is not None
    t = _ros_node.add_task(req.x, req.y, req.yaw)
    return {"ok": True, "task": asdict(t)}


@app.get("/state")
def state():
    assert _ros_node is not None
    return _ros_node.get_state()


def main():
    global _ros_node
    rclpy.init()
    _ros_node = FMSServer()

    # ROS spin thread
    def ros_spin():
        try:
            rclpy.spin(_ros_node)
        finally:
            _ros_node.shutdown()
            _ros_node.destroy_node()
            rclpy.shutdown()

    threading.Thread(target=ros_spin, daemon=True).start()

    # Web
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()

