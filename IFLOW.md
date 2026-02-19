# PDF Translate Online - 项目开发文档

## 项目概述

PDF Translate Online 是一个本地部署的 Web 应用程序，专为翻译英文学术 PDF 为中文而设计。采用分屏阅读模式，提供强大的同步滚动功能和实时翻译进度反馈。

**技术栈:**
- **前端**: React 19 + TypeScript + Vite
- **后端**: FastAPI + Python 3.13
- **任务队列**: Celery + Redis
- **容器化**: Docker + Docker Compose
- **OCR**: PaddleOCR
- **PDF 处理**: PyMuPDF (fitz)

## 核心特性

- **分屏阅读模式**: 左侧原文页 / 右侧译文页，强力同步滚动
- **智能优先级翻译**: 首次翻译前 3 页优先处理，其余页面后台进行
- **BYOK 模型配置**: 支持 OpenAI 及 OpenAI 兼容的翻译 API（默认支持 DeepSeek 和 Qwen）
- **自动故障转移**: 翻译失败时自动切换备用提供商
- **图片/图表 OCR 重绘**: 对 PDF 中的图片、图表文字进行 OCR 翻译重绘（尽力而为）
- **SSE 实时事件推送**: 使用 Server-Sent Events 提供页面级别的实时翻译进度
- **会话级数据管理**: 支持临时会话数据存储，自动过期清理
- **Docker 一键部署**: 通过 Docker Compose 快速启动完整服务栈

## 项目结构

```
pdfTranslate/
├── backend/                    # Python FastAPI 后端
│   ├── app/
│   │   ├── api/               # API 路由层
│   │   ├── core/              # 核心配置
│   │   ├── schemas/           # Pydantic 数据模型
│   │   ├── services/          # 业务逻辑层
│   │   ├── workers/           # Celery 任务定义
│   │   └── main.py            # FastAPI 应用入口
│   ├── requirements.txt       # Python 依赖
│   ├── .env.example           # 环境变量模板
│   └── celery_worker.py       # Celery Worker 启动脚本
│
├── frontend/                   # React 前端
│   ├── src/
│   │   ├── api.ts            # API 客户端
│   │   ├── App.tsx           # 主应用组件
│   │   ├── main.tsx          # 入口文件
│   │   ├── types.ts          # TypeScript 类型定义
│   │   └── style.css         # 全局样式
│   ├── package.json           # Node.js 依赖
│   ├── tsconfig.json          # TypeScript 配置
│   └── vite.config.ts         # Vite 配置
│
├── runtime-data/               # 运行时数据存储
│   └── {session_id}/          # 会话数据目录
│       ├── source.pdf         # 原始 PDF
│       ├── original/          # 原文页面图片
│       └── translated/        # 译文页面图片
│
├── _research/                  # 参考研究项目
│   ├── BabelDOC/             # 文档翻译参考
│   ├── PDFMathTranslate/     # 数学 PDF 翻译参考
│   └── PolyglotPDF/          # 多语言 PDF 翻译参考
│
├── docker-compose.yml          # Docker Compose 编排文件
└── README.md                   # 项目说明文档
```

## 快速开始

### 使用 Docker Compose 部署

这是推荐的部署方式，一键启动所有服务：

```bash
docker compose up --build
```

部署完成后：
- **Web UI**: http://localhost:5173
- **API**: http://localhost:8000
- **API 文档**: http://localhost:8000/docs

### 环境变量配置

在启动前，建议配置翻译 API 密钥。可在 `docker-compose.yml` 中的 `web` 服务配置环境变量：

| 环境变量 | 说明 | 默认值 |
|---------|------|--------|
| `VITE_DEFAULT_PRIMARY_API_KEY` | 主翻译提供商 API Key | - |
| `VITE_DEFAULT_BACKUP_API_KEY` | 备用翻译提供商 API Key | - |

后端配置（通过 `docker-compose.yml` 或 `backend/.env.example`）：

| 环境变量 | 说明 | 默认值 |
|---------|------|--------|
| `REDIS_URL` | Redis 连接地址 | `redis://redis:6379/0` |
| `STORAGE_ROOT` | 会话数据存储根目录 | `/tmp/pdftranslate/sessions` |
| `SESSION_TTL_MINUTES` | 会话过期时间（分钟） | `120` |
| `MAX_UPLOAD_MB` | 最大上传文件大小（MB） | `30` |
| `RENDER_DPI` | PDF 渲染 DPI | `160` |
| `HIGH_PRIORITY_QUEUE` | 高优先级队列名 | `page_high` |
| `NORMAL_PRIORITY_QUEUE` | 普通优先级队列名 | `page_normal` |
| `MAX_RETRIES_PER_PROVIDER` | 每个提供商最大重试次数 | `1` |
| `BATCH_SEGMENT_SIZE` | 批处理分段大小 | `36` |
| `BATCH_SEGMENT_CHAR_LIMIT` | 批处理字符限制 | `8000` |
| `TRANSLATION_TEMPERATURE` | 翻译温度参数 | `0.2` |
| `ENABLE_STRICT_LOW_QUALITY_RETRY` | 启用严格低质量重试 | `false` |
| `CLEAR_TRANSLATION_MEMORY_ON_RETRY` | 重试时清除翻译记忆 | `false` |

## API 接口

### 会话管理

#### 创建会话
```http
POST /v1/sessions
Content-Type: multipart/form-data

file: <pdf_file>
primary_provider: {"id": "deepseek-main", "model": "deepseek-chat", "base_url": "...", "api_key": "..."}
backup_provider: {"id": "qwen-backup", "model": "qwen-plus", "base_url": "...", "api_key": "..."}
```

#### 启动翻译
```http
POST /v1/sessions/{session_id}/start
```

#### 获取会话状态
```http
GET /v1/sessions/{session_id}/state
```

#### 订阅实时事件 (SSE)
```http
GET /v1/sessions/{session_id}/events
```

### 页面资源

#### 获取原文页面图片
```http
GET /v1/sessions/{session_id}/pages/{page_no}/original.png
```

#### 获取译文页面图片
```http
GET /v1/sessions/{session_id}/pages/{page_no}/translated.png
```

#### 重试页面翻译
```http
POST /v1/sessions/{session_id}/pages/{page_no}/retry
```

### 会话报告

#### 获取翻译报告
```http
GET /v1/sessions/{session_id}/report
```

#### 删除会话
```http
DELETE /v1/sessions/{session_id}
```

## 开发指南

### 前端开发

```bash
cd frontend

# 安装依赖
npm install

# 开发模式
npm run dev

# 构建
npm run build

# 预览构建结果
npm run preview
```

### 后端开发

```bash
cd backend

# 安装依赖
pip install -r requirements.txt

# 启动 API 服务
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 启动 Celery Worker
celery -A celery_worker.celery_app worker -Q page_high,page_normal --loglevel=info
```

### 代码规范

- **TypeScript**: 严格模式启用，使用 `@types/react` 和 `@types/react-dom`
- **Python**: 遵循 PEP 8，使用 Pydantic 进行数据验证
- **Git**: 使用 `.gitignore` 排除 `node_modules/`、`__pycache__/`、`runtime-data/` 等目录

## 架构说明

### 服务组件

1. **API 服务** (`backend/app/main.py`): FastAPI 应用，处理 HTTP 请求和 SSE 事件
2. **Worker 服务** (`backend/celery_worker.py`): Celery Worker，处理异步翻译任务
3. **Redis 服务**: 作为 Celery 的消息代理和结果存储
4. **Web 服务** (`frontend/`): React SPA，提供用户界面

### 数据流

1. 用户上传 PDF → 创建会话 → 存储 `source.pdf`
2. 提取原文页面 → 渲染为 PNG → 存储到 `original/`
3. 启动翻译 → 创建 Celery 任务 → 高优先级队列处理前 3 页
4. Worker 执行翻译 → OCR 识别 → API 调用 → 重绘 → 存储到 `translated/`
5. SSE 推送进度 → 前端更新 UI → 用户查看翻译结果

### 翻译策略

- **文本提取**: 使用 PyMuPDF 提取 PDF 文本和布局
- **图片 OCR**: 使用 PaddleOCR 识别图片中的文字
- **翻译 API**: 调用 OpenAI 兼容的翻译 API
- **重绘机制**: 将翻译后的文本重新绘制到原始 PDF 页面位置
- **质量重试**: 低质量翻译结果可自动或手动重试

## 注意事项

- 本应用**不提供文件下载**功能，设计为在线阅读器
- 会话数据是临时的，过期或手动删除后会自动清理
- OCR 质量取决于源图片清晰度和运行环境
- 翻译 API 需要用户自行配置，支持 OpenAI 兼容的任何提供商
- 建议使用 Docker 部署以确保环境一致性

## 故障排除

### Docker 启动失败

检查端口占用：
```bash
netstat -ano | findstr "5173 8000 6379"
```

### 翻译失败

1. 检查 API Key 配置是否正确
2. 查看 Docker 日志：`docker compose logs api worker`
3. 确认网络连接正常，可访问翻译 API

### OCR 效果差

1. 尝试提高 `RENDER_DPI` 参数（默认 160）
2. 确保源 PDF 图片清晰度足够
3. 检查 PaddleOCR 模型是否正确加载

## 参考资源

- [FastAPI 官方文档](https://fastapi.tiangolo.com/)
- [React 官方文档](https://react.dev/)
- [Celery 官方文档](https://docs.celeryq.dev/)
- [PyMuPDF 文档](https://pymupdf.readthedocs.io/)
- [PaddleOCR 文档](https://github.com/PaddlePaddle/PaddleOCR)