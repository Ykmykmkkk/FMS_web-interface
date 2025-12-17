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
from fastapi.responses import HTMLResponse
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
    # ✅ 맵/로봇 위치는 roslibjs가 직접 /map,/odom,/amcl_pose 구독해서 그린다.
    # ✅ 목표 전송은 2가지:
    #   - (A) FMS 큐로 보내기: /tasks (추천: “FMS가 제어한다” 보여줌)
    #   - (B) 브라우저에서 Nav2 액션 직접 보내기: /navigate_to_pose (원하면 사용)
    return """
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>FMS Dashboard + Map (rosbridge)</title>
    <script src="/static/roslib.min.js"></script>
    <style>
      body { font-family: sans-serif; margin: 16px; }
      #mapCanvas { border: 1px solid #333; background: #fff; display:block; margin-top:8px; }
      #pose { margin-top: 8px; }
      .row { display:flex; gap: 18px; flex-wrap: wrap; align-items:flex-start; }
      .card { border:1px solid #ccc; border-radius:8px; padding:12px 14px; }
      .goal-panel { margin-top: 10px; padding: 10px; border: 1px solid #ccc; border-radius: 6px; }
      .goal-panel input { width: 80px; margin-right: 8px; }
      table { border-collapse: collapse; margin-top: 8px; }
      td, th { border: 1px solid #ccc; padding: 6px 10px; font-size: 13px; }
      .muted { color:#666; font-size: 12px; }
      .warn { color:#b45309; }
    </style>
  </head>
  <body>
    <h2>FMS Web Panel (Single Robot)</h2>
    <div class="muted">Map/Pose는 rosbridge(9090)로 직접 구독해서 표시, Task는 FMS(FastAPI)가 큐로 실행</div>

    <p>ROS 연결 상태: <b><span id="status">연결 중...</span></b></p>

    <div class="row">
      <div class="card">
        <h3>Teleop (/cmd_vel)</h3>
        <button id="forwardBtn">앞으로</button>
        <button id="stopBtn">멈추기</button>
        <div class="muted">※ Nav2 주행 중엔 cmd_vel 충돌 가능</div>
      </div>

      <div class="card">
        <h3>FMS 상태</h3>
        <div id="fmsRobot"></div>
        <table>
          <thead><tr><th>ID</th><th>Status</th><th>Error</th></tr></thead>
          <tbody id="fmsTasks"></tbody>
        </table>
      </div>
    </div>

    <div class="row" style="margin-top:14px;">
      <div class="card">
        <h3>맵 & 로봇 위치</h3>
        <canvas id="mapCanvas" width="400" height="400"></canvas>
        <pre id="pose"></pre>
        <div class="muted">/map + /odom (+ /amcl_pose 있으면 보정)</div>
      </div>

      <div class="card" style="min-width:360px;">
        <h3>목표 이동</h3>

        <div class="goal-panel">
          <h4>A) FMS 큐로 작업 추가 (추천)</h4>
          <div>
            <label>X:</label><input type="number" id="taskX" step="0.1" value="0.0" />
            <label>Y:</label><input type="number" id="taskY" step="0.1" value="0.0" />
            <label>Yaw:</label><input type="number" id="taskYaw" step="0.1" value="0.0" />
            <button id="sendTaskBtn">작업 추가</button>
          </div>
          <pre id="taskStatus"></pre>
        </div>

        <div class="goal-panel">
          <h4>B) 브라우저에서 Nav2 goal 직접 전송 (옵션)</h4>
          <div>
            <label>X:</label><input type="number" id="goalX" step="0.1" value="0.0" />
            <label>Y:</label><input type="number" id="goalY" step="0.1" value="0.0" />
            <button id="sendGoalBtn">즉시 이동</button>
          </div>
          <pre id="goalStatus"></pre>
        </div>

      </div>
    </div>

    <script>
      // ===== 1. ROS 브리지 연결 =====
      const ros = new ROSLIB.Ros({ url: "ws://localhost:9090" });

      const statusEl = document.getElementById("status");
      const poseEl = document.getElementById("pose");
      const goalStatusEl = document.getElementById("goalStatus");
      const taskStatusEl = document.getElementById("taskStatus");

      ros.on("connection", () => { statusEl.textContent = "✅ 연결됨"; });
      ros.on("error", (error) => { statusEl.textContent = "❌ 오류"; console.error(error); });
      ros.on("close", () => { statusEl.textContent = "🔌 연결 종료"; });

      // ===== 2. 캔버스 / 맵 상태 =====
      const canvas = document.getElementById("mapCanvas");
      const ctx = canvas.getContext("2d");

      let mapInfo = null;
      let mapData = null;
      let mapImageReady = false;
      let cachedMapImage = null;

      let odomPose = null;     // odom frame
      let amclPose = null;     // map frame
      let poseOffset = null;   // map-odom offset
      let robotPose = null;    // map frame best estimate

      let targetRobotPixel = null;
      let currentRobotPixel = null;

      function drawMap() {
        if (!mapInfo || !mapData) return;

        const width = mapInfo.width;
        const height = mapInfo.height;
        const data = mapData;

        const SCALE = 2;
        canvas.width = width * SCALE;
        canvas.height = height * SCALE;

        const imageData = ctx.createImageData(canvas.width, canvas.height);
        const pixels = imageData.data;

        for (let y = 0; y < height; y++) {
          for (let x = 0; x < width; x++) {
            const idxMap = x + y * width;
            const occ = data[idxMap];

            let color;
            if (occ < 0) color = 220;
            else if (occ === 0) color = 255;
            else color = 0;

            const canvasX = x * SCALE;
            const canvasY = (height - 1 - y) * SCALE;

            for (let dy = 0; dy < SCALE; dy++) {
              for (let dx = 0; dx < SCALE; dx++) {
                const px = canvasX + dx;
                const py = canvasY + dy;
                const idx = (px + py * canvas.width) * 4;
                pixels[idx] = color;
                pixels[idx + 1] = color;
                pixels[idx + 2] = color;
                pixels[idx + 3] = 255;
              }
            }
          }
        }

        ctx.putImageData(imageData, 0, 0);
        cachedMapImage = imageData;
        mapImageReady = true;
      }
      const OFFSET_X = 2.0;
      const OFFSET_Y = 1.0;
      function robotPoseToPixel(pose) {
        if (!mapInfo) return null;

        const SCALE = 2;
        const res = mapInfo.resolution;
        const origin = mapInfo.origin;

        const rx = pose.x + OFFSET_X;
        const ry = pose.y + OFFSET_Y;
        const cellX = (rx - origin.position.x) / res;
        const cellY = (ry - origin.position.y) / res;

        const canvasX = cellX * SCALE;
        const canvasY = (mapInfo.height - 1 - cellY) * SCALE;

        return { x: canvasX, y: canvasY };
      }

      function drawRobotOnMap() {
        if (!mapImageReady || !currentRobotPixel) return;
        const { x, y } = currentRobotPixel;

        ctx.fillStyle = "#ff3b3b";
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fill();
      }

      // ===== 3. /map 구독 =====
      const mapListener = new ROSLIB.Topic({
        ros: ros,
        name: "/map",
        messageType: "nav_msgs/OccupancyGrid",
      });

      mapListener.subscribe((msg) => {
        mapInfo = msg.info;
        mapData = msg.data;
        drawMap();
      });

      // ===== 4. /amcl_pose 구독 =====
      const amclListener = new ROSLIB.Topic({
        ros: ros,
        name: "/amcl_pose",
        messageType: "geometry_msgs/PoseWithCovarianceStamped",
      });

      amclListener.subscribe((msg) => {
        const { x, y } = msg.pose.pose.position;
        amclPose = { x, y };
        if (odomPose) {
          poseOffset = { x: amclPose.x - odomPose.x, y: amclPose.y - odomPose.y };
        }
      });

      // ===== 5. /odom 구독 =====
      const odomListener = new ROSLIB.Topic({
        ros: ros,
        name: "/odom",
        messageType: "nav_msgs/Odometry",
      });

      odomListener.subscribe((msg) => {
        const { x, y } = msg.pose.pose.position;
        odomPose = { x, y };

        // 텍스트
        poseEl.textContent = `odom x: ${x.toFixed(2)}, y: ${y.toFixed(2)}`
          + (amclPose ? `\\namcl(map) x: ${amclPose.x.toFixed(2)}, y: ${amclPose.y.toFixed(2)}` : "");

        // map frame으로 보정(있으면)
        let mapPose;
        if (poseOffset) mapPose = { x: odomPose.x + poseOffset.x, y: odomPose.y + poseOffset.y };
        else if (amclPose) mapPose = { ...amclPose };
        else mapPose = { ...odomPose };

        robotPose = mapPose;

        const pixel = robotPoseToPixel(robotPose);
        if (pixel) {
          targetRobotPixel = pixel;
          if (!currentRobotPixel) currentRobotPixel = { ...targetRobotPixel };
        }
      });

      // ===== 6. 애니메이션 루프 =====
      function animationLoop() {
        if (mapImageReady && cachedMapImage) {
          ctx.putImageData(cachedMapImage, 0, 0);

          if (currentRobotPixel && targetRobotPixel) {
            const alpha = 0.2;
            currentRobotPixel.x += (targetRobotPixel.x - currentRobotPixel.x) * alpha;
            currentRobotPixel.y += (targetRobotPixel.y - currentRobotPixel.y) * alpha;
            drawRobotOnMap();
          }
        }
        requestAnimationFrame(animationLoop);
      }
      animationLoop();

      // ===== 7. /cmd_vel =====
      const cmdVel = new ROSLIB.Topic({
        ros: ros,
        name: "/cmd_vel",
        messageType: "geometry_msgs/Twist",
      });

      document.getElementById("forwardBtn").onclick = () => {
        cmdVel.publish(new ROSLIB.Message({
          linear: { x: 0.2, y: 0.0, z: 0.0 },
          angular: { x: 0.0, y: 0.0, z: 0.0 },
        }));
      };

      document.getElementById("stopBtn").onclick = () => {
        cmdVel.publish(new ROSLIB.Message({
          linear: { x: 0.0, y: 0.0, z: 0.0 },
          angular: { x: 0.0, y: 0.0, z: 0.0 },
        }));
      };

      // ===== 8-A. FMS 큐로 작업 추가 (FastAPI /tasks) =====
      document.getElementById("sendTaskBtn").onclick = async () => {
        const x = parseFloat(document.getElementById("taskX").value);
        const y = parseFloat(document.getElementById("taskY").value);
        const yaw = parseFloat(document.getElementById("taskYaw").value);

        if (Number.isNaN(x) || Number.isNaN(y) || Number.isNaN(yaw)) {
          taskStatusEl.textContent = "⚠️ x, y, yaw 값을 올바르게 입력하세요.";
          return;
        }

        taskStatusEl.textContent = "Sending task to FMS...";
        const res = await fetch("/tasks", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ x, y, yaw }),
        });

        if (!res.ok) {
          taskStatusEl.textContent = "❌ FMS task 추가 실패";
          return;
        }
        taskStatusEl.textContent = `✅ FMS 작업 추가됨: (${x.toFixed(2)}, ${y.toFixed(2)}, yaw=${yaw.toFixed(2)})`;
      };

      // ===== 8-B. Nav2 goal 직접 전송(옵션) =====
      const navActionClient = new ROSLIB.ActionClient({
        ros: ros,
        serverName: "/navigate_to_pose",
        actionName: "nav2_msgs/action/NavigateToPose",
      });

      document.getElementById("sendGoalBtn").onclick = () => {
        const x = parseFloat(document.getElementById("goalX").value);
        const y = parseFloat(document.getElementById("goalY").value);

        if (Number.isNaN(x) || Number.isNaN(y)) {
          goalStatusEl.textContent = "⚠️ x, y 값을 올바르게 입력하세요.";
          return;
        }

        const goal = new ROSLIB.Goal({
          actionClient: navActionClient,
          goalMessage: {
            pose: {
              header: { frame_id: "map" },
              pose: {
                position: { x: x, y: y, z: 0.0 },
                orientation: { x: 0.0, y: 0.0, z: 0.0, w: 1.0 },
              },
            },
          },
        });

        goalStatusEl.textContent = `📍 goal 전송: (${x.toFixed(2)}, ${y.toFixed(2)})`;

        goal.on("feedback", (_) => {
          goalStatusEl.textContent = "🚗 이동 중...";
        });

        goal.on("result", (res) => {
          goalStatusEl.textContent = "✅ 완료 처리됨 (result 수신)";
          console.log("Nav2 result:", res);
        });

        goal.send();
      };

      // ===== 9. FMS 상태 폴링 =====
      async function refreshFMS() {
        const r = await fetch("/state");
        const s = await r.json();

        const el = document.getElementById("fmsRobot");
        el.innerHTML = `Robot: <b>${s.robot.status}</b> | current_task: <b>${s.robot.current_task_id ?? ""}</b>`
          + (s.robot.last_error ? ` <span class="warn">| last_error: ${s.robot.last_error}</span>` : "");

        const tb = document.getElementById("fmsTasks");
        tb.innerHTML = "";
        s.tasks.forEach(t => {
          const tr = document.createElement("tr");
          tr.innerHTML = `<td>${t.task_id}</td><td>${t.status}</td><td>${t.error ?? ""}</td>`;
          tb.appendChild(tr);
        });
      }
      setInterval(refreshFMS, 800);
      refreshFMS();

    </script>
  </body>
</html>
"""


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

