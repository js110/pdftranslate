import { useEffect, useMemo, useState } from 'react'
import type { FormEvent } from 'react'
import {
  createSession,
  deleteSession,
  exportResultPdf,
  exportRewritePdf,
  getState,
  getRewriteState,
  retryPage,
  saveResultPdf,
  startRewrite,
  startSession,
  subscribeEvents,
} from './api'
import { ProviderControls } from './components/ProviderControls'
import { SessionActions } from './components/SessionActions'
import { UploadPanel } from './components/UploadPanel'
import { ViewerGrid } from './components/ViewerGrid'
import {
  DEFAULT_PRIMARY_API_KEY,
  DEFAULT_PRIMARY_BASE_URL,
  DEFAULT_PRIMARY_ID,
  DEFAULT_PRIMARY_MODEL,
  LAST_SESSION_STORAGE,
  PRIMARY_KEY_STORAGE,
  isMissingSessionError,
  mergeCacheVersionFromState,
} from './constants'
import { inferPresetKey, inferTimeout } from './providers'
import type { ProviderPresetKey } from './providers'
import type {
  EventEnvelope,
  RewriteState,
  SessionCreateResponse,
  SessionState,
  StartSessionRequest,
} from './types'

function readLocalStorage(key: string): string {
  if (typeof window === 'undefined') return ''
  return window.localStorage.getItem(key) ?? ''
}

function App() {
  const [file, setFile] = useState<File | null>(null)
  const [session, setSession] = useState<SessionCreateResponse | null>(null)
  const [state, setState] = useState<SessionState | null>(null)
  const [restoringSession, setRestoringSession] = useState(true)
  const [loading, setLoading] = useState(false)
  const [starting, setStarting] = useState(false)
  const [savingPdf, setSavingPdf] = useState(false)
  const [exportingPdf, setExportingPdf] = useState(false)
  const [rewrite, setRewrite] = useState<RewriteState | null>(null)
  const [rewriteStarting, setRewriteStarting] = useState(false)
  const [exportingRewrite, setExportingRewrite] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [info, setInfo] = useState<string | null>(null)
  const [cacheVersion, setCacheVersion] = useState<Record<number, number>>({})
  const [controlsOpen, setControlsOpen] = useState(true)

  const [primaryId, setPrimaryId] = useState(DEFAULT_PRIMARY_ID)
  const [primaryModel, setPrimaryModel] = useState(DEFAULT_PRIMARY_MODEL)
  const [primaryBaseUrl, setPrimaryBaseUrl] = useState(DEFAULT_PRIMARY_BASE_URL)
  const [primaryKey, setPrimaryKey] = useState(() => {
    const local = readLocalStorage(PRIMARY_KEY_STORAGE).trim()
    return DEFAULT_PRIMARY_API_KEY || local
  })
  const [primaryPreset, setPrimaryPreset] = useState<ProviderPresetKey>(() =>
    inferPresetKey(DEFAULT_PRIMARY_BASE_URL, DEFAULT_PRIMARY_MODEL),
  )

  const sortedPages = useMemo(() => {
    if (!state) return []
    return [...state.page_states].sort((a, b) => a.page_no - b.page_no)
  }, [state])

  const progress = useMemo(() => {
    const total = sortedPages.length
    let ready = 0
    let processing = 0
    let pending = 0
    let failed = 0
    for (const page of sortedPages) {
      if (page.status === 'ready') ready += 1
      if (page.status === 'processing') processing += 1
      if (page.status === 'pending') pending += 1
      if (page.status === 'failed') failed += 1
    }
    return { total, ready, processing, pending, failed }
  }, [sortedPages])

  const refreshState = async (sessionId: string) => {
    const nextState = await getState(sessionId)
    setState(nextState)
    setCacheVersion((prev) => mergeCacheVersionFromState(prev, nextState))
  }

  const refreshRewrite = async (sessionId: string) => {
    try {
      setRewrite(await getRewriteState(sessionId))
    } catch {
      // session gone — ignore, state sync will surface it
    }
  }

  useEffect(() => {
    let cancelled = false
    const restoreSession = async () => {
      if (typeof window === 'undefined') {
        setRestoringSession(false)
        return
      }

      const savedSessionId = (window.localStorage.getItem(LAST_SESSION_STORAGE) ?? '').trim()
      if (!savedSessionId) {
        setRestoringSession(false)
        return
      }

      try {
        const nextState = await getState(savedSessionId)
        if (cancelled) return
        setSession({
          session_id: nextState.session_id,
          page_count: nextState.page_count,
          expires_at: nextState.expires_at,
        })
        setState(nextState)
        setCacheVersion((prev) => mergeCacheVersionFromState(prev, nextState))
        void refreshRewrite(savedSessionId)
      } catch (err) {
        if (!cancelled && isMissingSessionError(err)) {
          window.localStorage.removeItem(LAST_SESSION_STORAGE)
        } else if (!cancelled) {
          setError('\u6062\u590d\u4e0a\u6b21\u4f1a\u8bdd\u5931\u8d25\uff0c\u8bf7\u68c0\u67e5\u540e\u7aef\u8fde\u63a5\u540e\u5237\u65b0\u91cd\u8bd5\u3002')
        }
      } finally {
        if (!cancelled) setRestoringSession(false)
      }
    }

    void restoreSession()
    return () => {
      cancelled = true
    }
  }, [])

  const onUpload = async (event: FormEvent) => {
    event.preventDefault()
    if (!file) {
      setError('\u8bf7\u5148\u9009\u62e9 PDF \u6587\u4ef6\u3002')
      return
    }

    setLoading(true)
    setError(null)
    setInfo(null)
    try {
      const created = await createSession(file)
      setSession(created)
      if (typeof window !== 'undefined') {
        window.localStorage.setItem(LAST_SESSION_STORAGE, created.session_id)
      }
      await refreshState(created.session_id)
      await refreshRewrite(created.session_id)
    } catch (err) {
      setError(err instanceof Error ? err.message : '\u4e0a\u4f20\u5931\u8d25')
    } finally {
      setLoading(false)
    }
  }

  const onStart = async () => {
    if (!session) return
    if (state && ['running', 'ready'].includes(state.overall_status)) {
      setError('\u5f53\u524d\u4f1a\u8bdd\u5df2\u5728\u7ffb\u8bd1\u4e2d\uff0c\u65e0\u9700\u91cd\u590d\u70b9\u51fb\u3002')
      return
    }
    if (!primaryKey.trim()) {
      setError('\u4e3b\u6a21\u578b API Key \u4e0d\u80fd\u4e3a\u7a7a\u3002')
      return
    }

    const payload: StartSessionRequest = {
      primary_provider: {
        id: primaryId,
        model: primaryModel,
        base_url: primaryBaseUrl.trim() || undefined,
        api_key: primaryKey,
        timeout_sec: inferTimeout(primaryPreset),
      },
      style_profile: 'academic_conservative',
    }

    setStarting(true)
    setError(null)
    setInfo(null)
    try {
      await startSession(session.session_id, payload)
      await refreshState(session.session_id)
    } catch (err) {
      const message = err instanceof Error ? err.message : '\u542f\u52a8\u7ffb\u8bd1\u5931\u8d25'
      if (message.includes('Session already started')) {
        setError('\u5f53\u524d\u4f1a\u8bdd\u5df2\u5728\u7ffb\u8bd1\u4e2d\uff0c\u65e0\u9700\u91cd\u590d\u70b9\u51fb\u3002')
      } else {
        setError(message)
      }
    } finally {
      setStarting(false)
    }
  }

  useEffect(() => {
    if (typeof window === 'undefined') return
    window.localStorage.setItem(PRIMARY_KEY_STORAGE, primaryKey)
  }, [primaryKey])

  useEffect(() => {
    if (typeof window === 'undefined') return
    if (!session?.session_id) return
    window.localStorage.setItem(LAST_SESSION_STORAGE, session.session_id)
  }, [session?.session_id])

  useEffect(() => {
    if (!session) return

    const source = subscribeEvents(session.session_id, (event: EventEnvelope) => {
      if (event.event === 'page_ready') {
        const pageNo = Number(event.payload.page_no)
        setCacheVersion((prev) => ({ ...prev, [pageNo]: (prev[pageNo] ?? 0) + 1 }))
      }
      if (event.event === 'rewrite_progress') {
        setRewrite((prev) =>
          prev
            ? {
                ...prev,
                status: 'running',
                stage: String(event.payload.stage ?? prev.stage ?? ''),
                done: Number(event.payload.done ?? 0),
                total: Number(event.payload.total ?? prev.total ?? 0),
                error: null,
              }
            : prev,
        )
      }
      if (event.event === 'rewrite_ready' || event.event === 'rewrite_failed') {
        void refreshRewrite(session.session_id)
      }
      void refreshState(session.session_id)
    })

    return () => {
      source.close()
    }
  }, [session?.session_id])

  useEffect(() => {
    if (!session || !state || !['running', 'created'].includes(state.overall_status)) return
    const timer = window.setInterval(() => {
      void refreshState(session.session_id)
    }, 3000)
    return () => window.clearInterval(timer)
  }, [session?.session_id, state?.overall_status])

  useEffect(() => {
    if (!session || rewrite?.status !== 'running') return
    const timer = window.setInterval(() => {
      void refreshRewrite(session.session_id)
    }, 3000)
    return () => window.clearInterval(timer)
  }, [session?.session_id, rewrite?.status])

  useEffect(() => {
    if (!state || !error) return
    if (error.includes('\u5df2\u5728\u7ffb\u8bd1\u4e2d') && state.overall_status === 'running') {
      setError(null)
    }
  }, [state?.overall_status, error])

  useEffect(() => {
    if (!info) return
    const t = setTimeout(() => setInfo(null), 5000)
    return () => clearTimeout(t)
  }, [info])

  useEffect(() => {
    if (state?.overall_status === 'running') {
      setControlsOpen(false)
    }
  }, [state?.overall_status])

  const destroySession = async () => {
    if (!session) return
    await deleteSession(session.session_id)
    if (typeof window !== 'undefined') {
      window.localStorage.removeItem(LAST_SESSION_STORAGE)
    }
    setSession(null)
    setState(null)
    setRewrite(null)
    setCacheVersion({})
    setError(null)
    setInfo(null)
  }

  const onSaveResultPdf = async () => {
    if (!session) return
    setSavingPdf(true)
    setError(null)
    setInfo(null)
    try {
      const saved = await saveResultPdf(session.session_id)
      setInfo(`结果 PDF 已保存: ${saved.saved_path}（译文页 ${saved.translated_pages}/${saved.page_count}）`)
    } catch (err) {
      setError(err instanceof Error ? err.message : '保存 PDF 失败')
    } finally {
      setSavingPdf(false)
    }
  }

  const onExportResultPdf = async () => {
    if (!session) return
    setExportingPdf(true)
    setError(null)
    setInfo(null)
    try {
      await exportResultPdf(session.session_id)
      setInfo('\u5bfc\u51fa\u7ed3\u679c PDF \u4e0b\u8f7d\u5df2\u5f00\u59cb')
    } catch (err) {
      setError(err instanceof Error ? err.message : '\u5bfc\u51fa PDF \u5931\u8d25')
    } finally {
      setExportingPdf(false)
    }
  }

  const onRewrite = async () => {
    if (!session) return
    if (rewrite?.status === 'running') return
    if (!primaryKey.trim()) {
      setError('主模型 API Key 不能为空。')
      return
    }
    const payload: StartSessionRequest = {
      primary_provider: {
        id: primaryId,
        model: primaryModel,
        base_url: primaryBaseUrl.trim() || undefined,
        api_key: primaryKey,
        timeout_sec: inferTimeout(primaryPreset),
      },
      style_profile: 'academic_conservative',
    }
    setRewriteStarting(true)
    setError(null)
    setInfo(null)
    try {
      setRewrite(await startRewrite(session.session_id, payload))
      setInfo('原位翻译已启动：将保留原 PDF 样式输出可搜索的中文 PDF。')
    } catch (err) {
      setError(err instanceof Error ? err.message : '启动原位翻译失败')
    } finally {
      setRewriteStarting(false)
    }
  }

  const onExportRewritePdf = async () => {
    if (!session) return
    setExportingRewrite(true)
    setError(null)
    setInfo(null)
    try {
      await exportRewritePdf(session.session_id)
      setInfo('原位翻译 PDF 下载已开始')
    } catch (err) {
      setError(err instanceof Error ? err.message : '下载原位翻译 PDF 失败')
    } finally {
      setExportingRewrite(false)
    }
  }

  const onRetryPage = async (pageNo: number) => {
    if (!session) return
    setError(null)
    setInfo(null)
    try {
      await retryPage(session.session_id, pageNo)
      await refreshState(session.session_id)
    } catch (err) {
      setError(err instanceof Error ? err.message : `重试第 ${pageNo} 页失败`)
    }
  }

  const startBlocked = starting || !session || !!(state && ['running', 'ready'].includes(state.overall_status))
  const startLabel = starting
    ? '\u542f\u52a8\u4e2d...'
    : state?.overall_status === 'running'
      ? '\u7ffb\u8bd1\u8fdb\u884c\u4e2d...'
      : state?.overall_status === 'ready'
        ? '\u5df2\u5b8c\u6210'
        : '\u5f00\u59cb\u7ffb\u8bd1\uff08\u524d 3 \u9875\u4f18\u5148\uff09'

  return (
    <div className="app-shell">
      <header className="header-bar">
        <h1>PDF {'\u79d1\u7814\u7ffb\u8bd1\u5668'}</h1>
        <span className="header-subtitle">{'\u5de6\u4fa7\u539f\u6587 / \u53f3\u4fa7\u8bd1\u6587\uff0c\u5f3a\u540c\u6b65\u6eda\u52a8'}</span>
      </header>

      {!session && restoringSession && (
        <section className="upload-stage">
          <div className="upload-panel panel">
            <h2>{'\u6b63\u5728\u6062\u590d\u4e0a\u6b21\u4f1a\u8bdd...'}</h2>
          </div>
        </section>
      )}

      {!session && !restoringSession && (
        <UploadPanel file={file} setFile={setFile} loading={loading} onUpload={onUpload} />
      )}

      {session && (
        <section className="panel controls-panel">
          <div className="controls-panel-header">
            <button
              type="button"
              className="controls-toggle"
              onClick={() => setControlsOpen((prev) => !prev)}
            >
              {controlsOpen ? '收起设置 ▲' : '展开设置 ▼'}
            </button>
          </div>
          {controlsOpen && (
            <ProviderControls
              primaryPreset={primaryPreset}
              setPrimaryPreset={setPrimaryPreset}
              primaryId={primaryId}
              setPrimaryId={setPrimaryId}
              primaryModel={primaryModel}
              setPrimaryModel={setPrimaryModel}
              primaryBaseUrl={primaryBaseUrl}
              setPrimaryBaseUrl={setPrimaryBaseUrl}
              primaryKey={primaryKey}
              setPrimaryKey={setPrimaryKey}
            />
          )}

          <SessionActions
            state={state}
            progress={progress}
            startLabel={startLabel}
            startBlocked={startBlocked}
            savingPdf={savingPdf}
            exportingPdf={exportingPdf}
            rewrite={rewrite}
            rewriteStarting={rewriteStarting}
            exportingRewrite={exportingRewrite}
            onStart={onStart}
            onSaveResultPdf={onSaveResultPdf}
            onExportResultPdf={onExportResultPdf}
            onRewrite={onRewrite}
            onExportRewritePdf={onExportRewritePdf}
            onDestroySession={destroySession}
          />
        </section>
      )}

      {error && <section className="panel error-panel">{error}</section>}
      {info && <section className="panel info-panel">{info}</section>}

      {session && state && (
        <ViewerGrid session={session} state={state} cacheVersion={cacheVersion} onRetryPage={onRetryPage} />
      )}
    </div>
  )
}

export default App
