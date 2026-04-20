"""
Web Monitor สำหรับดูสถานะการทำงานแบบ Real-time
เข้าดูได้ที่ http://localhost:5555
"""

import threading
import time
import json
from datetime import datetime
from flask import Flask, render_template_string, jsonify
from flask_socketio import SocketIO

# ========================================
# HTML Template
# ========================================
MONITOR_HTML = """
<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🔍 SUT RF-DETR Monitor</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.5/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, sans-serif;
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            color: #fff;
            min-height: 100vh;
        }
        .header {
            background: rgba(255,255,255,0.05);
            backdrop-filter: blur(10px);
            padding: 20px 40px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 { font-size: 24px; }
        .header h1 span { color: #00d4ff; }
        .status-badge {
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: bold;
        }
        .status-running { background: #00c853; color: #000; }
        .status-stopped { background: #ff5252; color: #fff; }
        .container { padding: 30px 40px; }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .card {
            background: rgba(255,255,255,0.05);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 16px;
            padding: 24px;
            transition: transform 0.2s;
        }
        .card:hover { transform: translateY(-4px); }
        .card-title {
            font-size: 13px;
            color: #aaa;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }
        .card-value {
            font-size: 36px;
            font-weight: bold;
            color: #00d4ff;
        }
        .card-sub { font-size: 12px; color: #888; margin-top: 4px; }
        .log-section {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 16px;
            padding: 24px;
        }
        .log-section h2 { margin-bottom: 16px; font-size: 18px; }
        .log-list {
            max-height: 400px;
            overflow-y: auto;
            font-family: 'Consolas', monospace;
            font-size: 13px;
        }
        .log-item {
            padding: 8px 12px;
            border-bottom: 1px solid rgba(255,255,255,0.05);
            display: flex;
            justify-content: space-between;
        }
        .log-item:nth-child(even) { background: rgba(255,255,255,0.02); }
        .log-time { color: #888; }
        .log-event { color: #00d4ff; }
        .log-count { color: #00c853; font-weight: bold; }
        .class-bar {
            display: flex;
            align-items: center;
            margin: 6px 0;
        }
        .class-name {
            width: 120px;
            font-size: 13px;
            color: #ccc;
        }
        .class-bar-fill {
            height: 20px;
            background: linear-gradient(90deg, #00d4ff, #7c4dff);
            border-radius: 4px;
            transition: width 0.5s;
            margin-right: 10px;
        }
        .class-count { font-size: 13px; color: #00d4ff; }
        .passing-section {
            margin-top: 20px;
        }
        .pass-in { color: #00c853; font-size: 28px; font-weight: bold; }
        .pass-out { color: #ff5252; font-size: 28px; font-weight: bold; }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-thumb {
            background: rgba(255,255,255,0.2);
            border-radius: 3px;
        }
        .chart-container {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 16px;
            padding: 24px;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 <span>SUT</span> RF-DETR Monitor</h1>
        <div id="statusBadge" class="status-badge status-stopped">⏹ STOPPED</div>
    </div>

    <div class="container">
        <div class="grid">
            <div class="card">
                <div class="card-title">📷 Source</div>
                <div class="card-value" id="source" style="font-size:20px;">-</div>
                <div class="card-sub" id="modelInfo">Model: -</div>
            </div>
            <div class="card">
                <div class="card-title">⚡ FPS</div>
                <div class="card-value" id="fps">0</div>
                <div class="card-sub" id="frameCount">Frames: 0</div>
            </div>
            <div class="card">
                <div class="card-title">🎯 Total Objects</div>
                <div class="card-value" id="totalObjects">0</div>
                <div class="card-sub" id="classCount">Classes: 0</div>
            </div>
            <div class="card">
                <div class="card-title">✅ Passing Count</div>
                <div class="card-value">
                    <span class="pass-in" id="passIn">0</span>
                    <span style="color:#666;font-size:20px;"> IN </span>
                    <span class="pass-out" id="passOut">0</span>
                    <span style="color:#666;font-size:20px;"> OUT</span>
                </div>
                <div class="card-sub" id="passTotal">Total passed: 0</div>
            </div>
        </div>

        <div class="grid">
            <div class="card">
                <div class="card-title">📊 Class Distribution</div>
                <div id="classDistribution">
                    <p style="color:#666;">Waiting for data...</p>
                </div>
            </div>
            <div class="log-section">
                <h2>📋 Event Log</h2>
                <div class="log-list" id="logList">
                    <div class="log-item">
                        <span class="log-time">--:--:--</span>
                        <span>Waiting for connection...</span>
                    </div>
                </div>
            </div>
        </div>

        <div class="chart-container">
            <canvas id="timeChart" height="100"></canvas>
        </div>
    </div>

    <script>
        const socket = io();
        const maxDataPoints = 50;
        const chartData = { labels: [], objects: [], fps: [] };

        const ctx = document.getElementById('timeChart').getContext('2d');
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: chartData.labels,
                datasets: [
                    {
                        label: 'Objects',
                        data: chartData.objects,
                        borderColor: '#00d4ff',
                        backgroundColor: 'rgba(0,212,255,0.1)',
                        fill: true, tension: 0.4
                    },
                    {
                        label: 'FPS',
                        data: chartData.fps,
                        borderColor: '#00c853',
                        backgroundColor: 'rgba(0,200,83,0.1)',
                        fill: true, tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    x: { ticks: { color: '#666' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    y: { ticks: { color: '#666' }, grid: { color: 'rgba(255,255,255,0.05)' } }
                },
                plugins: { legend: { labels: { color: '#ccc' } } }
            }
        });

        socket.on('update', function(data) {
            // Status
            const badge = document.getElementById('statusBadge');
            badge.className = 'status-badge status-running';
            badge.textContent = '▶ RUNNING';

            // Cards
            document.getElementById('source').textContent = data.source || '-';
            document.getElementById('modelInfo').textContent = 'Model: ' + (data.model || '-');
            document.getElementById('fps').textContent = (data.fps || 0).toFixed(1);
            document.getElementById('frameCount').textContent = 'Frames: ' + (data.frame_count || 0);
            document.getElementById('totalObjects').textContent = data.total_objects || 0;
            document.getElementById('classCount').textContent = 'Classes: ' + (data.unique_classes || 0);
            document.getElementById('passIn').textContent = data.pass_in || 0;
            document.getElementById('passOut').textContent = data.pass_out || 0;
            document.getElementById('passTotal').textContent = 'Total passed: ' + ((data.pass_in||0) + (data.pass_out||0));

            // Class Distribution
            const distDiv = document.getElementById('classDistribution');
            const classDist = data.class_distribution || {};
            const maxCount = Math.max(...Object.values(classDist), 1);
            let html = '';
            for (const [cls, count] of Object.entries(classDist)) {
                const width = (count / maxCount * 100).toFixed(0);
                html += `<div class="class-bar">
                    <span class="class-name">${cls}</span>
                    <div class="class-bar-fill" style="width:${width}%"></div>
                    <span class="class-count">${count}</span>
                </div>`;
            }
            distDiv.innerHTML = html || '<p style="color:#666;">No detections</p>';

            // Chart
            const now = new Date().toLocaleTimeString();
            chartData.labels.push(now);
            chartData.objects.push(data.total_objects || 0);
            chartData.fps.push(data.fps || 0);
            if (chartData.labels.length > maxDataPoints) {
                chartData.labels.shift();
                chartData.objects.shift();
                chartData.fps.shift();
            }
            chart.update('none');
        });

        socket.on('log', function(data) {
            const logList = document.getElementById('logList');
            const item = document.createElement('div');
            item.className = 'log-item';
            item.innerHTML = `
                <span class="log-time">${data.time}</span>
                <span class="log-event">${data.event}</span>
                <span class="log-count">${data.detail || ''}</span>
            `;
            logList.insertBefore(item, logList.firstChild);
            if (logList.children.length > 100) logList.removeChild(logList.lastChild);
        });

        socket.on('stopped', function() {
            const badge = document.getElementById('statusBadge');
            badge.className = 'status-badge status-stopped';
            badge.textContent = '⏹ STOPPED';
        });
    </script>
</body>
</html>
"""


class WebMonitor:
    """Web-based monitor ทำงานเบื้องหลัง"""

    def __init__(self, port=5555):
        self.port = port
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'sut_rfdetr_secret'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode='threading')
        self._thread = None
        self._running = False

        @self.app.route('/')
        def index():
            return render_template_string(MONITOR_HTML)

        @self.app.route('/api/status')
        def api_status():
            return jsonify({"status": "running" if self._running else "stopped"})

    def start(self):
        """เริ่ม Web Monitor ใน background thread"""
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(
            target=lambda: self.socketio.run(
                self.app, host='0.0.0.0', port=self.port,
                debug=False, use_reloader=False, allow_unsafe_werkzeug=True
            ),
            daemon=True
        )
        self._thread.start()
        time.sleep(1)
        print(f"🌐 Web Monitor: http://localhost:{self.port}")

    def stop(self):
        self._running = False
        self.socketio.emit('stopped', {})

    def send_update(self, data: dict):
        """ส่งข้อมูลอัปเดตไปที่หน้าเว็บ"""
        try:
            self.socketio.emit('update', data)
        except Exception:
            pass

    def send_log(self, event: str, detail: str = ""):
        """ส่ง log event ไปที่หน้าเว็บ"""
        try:
            self.socketio.emit('log', {
                'time': datetime.now().strftime('%H:%M:%S'),
                'event': event,
                'detail': detail
            })
        except Exception:
            pass