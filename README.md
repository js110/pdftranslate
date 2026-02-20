# PDF Translate Online（双栏阅读器）

一个本地 Docker 部署的 Web 应用，用于将英文科研 PDF 翻译为中文，并提供左原文/右译文的同步阅读体验。

## 功能特性

- 单 PDF 会话上传
- 左原文 / 右译文双栏阅读（强同步滚动）
- 前 3 页优先翻译，其余页面后台继续处理
- BYOK 模型配置（OpenAI + OpenAI 兼容接口）
- 主备模型自动切换
- 图片/图表文本 OCR + 回写重绘（尽力而为）
- SSE 实时推送页面翻译进度
- 会话级数据隔离与自动清理

## Docker 启动

```bash
docker compose up --build
```

启动后访问：

- Web UI：`http://localhost:5173`
- API：`http://localhost:8000`

## 给他人使用（无需 Python/Node 开发环境）

### 方案 A（推荐）：你集中部署，别人只访问网页

- 在一台机器（服务器或局域网主机）上部署本项目
- 让他人访问 `http://<你的主机IP>:5173`
- 对方只需要浏览器

### 方案 B：对方本地运行 Docker Desktop

- 安装 Docker Desktop
- 解压/复制本项目目录
- 双击 `start.bat`
- 打开 `http://localhost:5173`

停止服务：`stop.bat`  
重启服务：`restart.bat`

## 开发模式热更新

使用 dev 覆盖配置启动（API、Worker、前端开发服务器热更新）：

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build redis api worker web-dev
```

开发模式行为：

- `api`：以 `uvicorn --reload` 方式运行，并挂载 `./backend`
- `worker`：Python 文件变化后自动重启 Celery Worker
- `web-dev`：运行 Vite Dev Server，开启 HMR 与轮询（适配 Docker Desktop）

## API 概览

- `POST /v1/sessions`
- `POST /v1/sessions/{session_id}/start`
- `GET /v1/sessions/{session_id}/state`
- `GET /v1/sessions/{session_id}/events`
- `GET /v1/sessions/{session_id}/pages/{page_no}/original.png`
- `GET /v1/sessions/{session_id}/pages/{page_no}/translated.png`
- `GET /v1/sessions/{session_id}/report`
- `POST /v1/sessions/{session_id}/pages/{page_no}/retry`
- `DELETE /v1/sessions/{session_id}`

## 说明

- 本项目按设计不提供“直接下载译文文件”的在线出口
- 会话数据是临时数据，过期或手动删除后会清理
- OCR 效果受源 PDF 图片清晰度与运行环境影响
