# PDF Translate Online（双栏阅读器）

一个本地 Docker 部署的 Web 应用，用于将英文科研 PDF 翻译为中文，并提供左原文/右译文的同步阅读体验。

## 效果演示视频

- 标题：`计算机专业科研论文翻译器`
- 地址：`https://www.bilibili.com/video/BV16aZeBrE5X/`
- 时长：`6分47秒`

[![PDF Translate 在线效果演示](https://i0.hdslb.com/bfs/archive/e6a26b38a1b27c808a6a3e4218439893a5acde13.jpg)](https://www.bilibili.com/video/BV16aZeBrE5X/)

点击封面可直接打开 B 站视频介绍页。

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

## API Key 配置（必做）

### 方式一：在网页中填写（推荐）

1. 启动服务后打开 `http://localhost:5173`
2. 上传 PDF
3. 在页面顶部填写主模型配置：
   - `主模型 Base URL`（例如 OpenAI 兼容地址）
   - `主模型名称`
   - `主模型 API Key`
4. 可选开启备用模型，并填写备用模型的 `Base URL / 模型名称 / API Key`
5. 点击“开始翻译”

### 方式二：启动前预置默认 Key（可选）

在项目根目录新建 `.env`，填入：

```env
VITE_DEFAULT_PRIMARY_API_KEY=你的主模型Key
VITE_DEFAULT_BACKUP_API_KEY=你的备用模型Key
```

然后启动：

```bash
docker compose up --build
```

说明：

- 页面里仍可手动修改模型参数与 Key
- 若你只用一个模型，备用 Key 可留空
- 不要把 API Key 提交到 Git 仓库或发到公开聊天记录中

### 三家模型 API Key 获取方式（官方）

#### 1. DeepSeek

- 平台首页：`https://platform.deepseek.com/`
- Key 管理页：`https://platform.deepseek.com/api_keys`
- 官方文档：`https://api-docs.deepseek.com/zh-cn/`

操作步骤：

1. 注册并登录 DeepSeek 平台
2. 进入 `API Keys` 页面，新建一个 Key 并复制保存
3. 在本项目页面中填写：
   - `Base URL`：`https://api.deepseek.com/v1`
   - `模型名称`：如 `deepseek-chat`
   - `API Key`：刚创建的 Key

#### 2. 腾讯混元（Tencent Hunyuan）

- API Key 管理文档：`https://cloud.tencent.com/document/product/1729/111008`
- OpenAI 兼容接口文档：`https://cloud.tencent.com/document/product/1729/111007`
- 控制台入口：`https://console.cloud.tencent.com/hunyuan/start`

操作步骤：

1. 登录腾讯云账号（如页面提示，先完成实名认证）
2. 开通混元服务并进入控制台
3. 按官方文档创建 API Key
4. 在本项目页面中填写：
   - `Base URL`：`https://api.hunyuan.cloud.tencent.com/v1`
   - `模型名称`：如 `hunyuan-turbos-latest`
   - `API Key`：控制台创建的 Key

#### 3. 阿里云通义千问（DashScope / 百炼）

- API Key 获取文档：`https://help.aliyun.com/zh/model-studio/get-api-key`
- Model Studio 控制台：`https://bailian.console.aliyun.com/`

操作步骤：

1. 登录阿里云并开通 Model Studio（百炼）
2. 按官方文档创建 API Key
3. 在本项目页面中填写：
   - `Base URL`（中国站）：`https://dashscope.aliyuncs.com/compatible-mode/v1`
   - `模型名称`：如 `qwen-plus`
   - `API Key`：刚创建的 Key

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
