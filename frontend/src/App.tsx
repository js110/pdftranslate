import { useEffect, useMemo, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import {
  createSession,
  deleteSession,
  ensurePage,
  getState,
  originalPageUrl,
  retryPage,
  saveResultPdf,
  startSession,
  subscribeEvents,
  translatedPageUrl,
} from './api'
import type { EventEnvelope, SessionCreateResponse, SessionState, StartSessionRequest } from './types'

const PRIMARY_KEY_STORAGE = 'pdftranslate_primary_api_key'
const BACKUP_KEY_STORAGE = 'pdftranslate_backup_api_key'
const LAST_SESSION_STORAGE = 'pdftranslate_last_session_id'
const DEFAULT_PRIMARY_ID = (import.meta.env.VITE_DEFAULT_PRIMARY_ID as string | undefined) ?? 'deepseek-main'
const DEFAULT_PRIMARY_MODEL = (import.meta.env.VITE_DEFAULT_PRIMARY_MODEL as string | undefined) ?? 'deepseek-chat'
const DEFAULT_PRIMARY_BASE_URL =
  (import.meta.env.VITE_DEFAULT_PRIMARY_BASE_URL as string | undefined) ?? 'https://api.deepseek.com/v1'
const DEFAULT_PRIMARY_API_KEY = (import.meta.env.VITE_DEFAULT_PRIMARY_API_KEY as string | undefined) ?? ''
const DEFAULT_BACKUP_ID = (import.meta.env.VITE_DEFAULT_BACKUP_ID as string | undefined) ?? 'deepseek-backup'
const DEFAULT_BACKUP_MODEL = (import.meta.env.VITE_DEFAULT_BACKUP_MODEL as string | undefined) ?? 'deepseek-chat'
const DEFAULT_BACKUP_BASE_URL =
  (import.meta.env.VITE_DEFAULT_BACKUP_BASE_URL as string | undefined) ?? 'https://api.deepseek.com/v1'
const DEFAULT_BACKUP_API_KEY = (import.meta.env.VITE_DEFAULT_BACKUP_API_KEY as string | undefined) ?? ''

type ProviderPresetKey = 'deepseek' | 'tencent' | 'aliyun' | 'custom'

type ProviderPreset = {
  key: ProviderPresetKey
  label: string
  id: string
  model: string
  baseUrl: string
}

const PROVIDER_PRESETS: ProviderPreset[] = [
  {
    key: 'deepseek',
    label: 'DeepSeek',
    id: 'deepseek-main',
    model: 'deepseek-chat',
    baseUrl: 'https://api.deepseek.com/v1',
  },
  {
    key: 'tencent',
    label: '腾讯混元',
    id: 'hunyuan-main',
    model: 'hunyuan-turbos-latest',
    baseUrl: 'https://api.hunyuan.cloud.tencent.com/v1',
  },
  {
    key: 'aliyun',
    label: '阿里通义千问',
    id: 'qwen-main',
    model: 'qwen-plus',
    baseUrl: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
  },
  {
    key: 'custom',
    label: '自定义（OpenAI 兼容）',
    id: 'custom-main',
    model: '',
    baseUrl: '',
  },
]

function inferPresetKey(baseUrl: string, model: string): ProviderPresetKey {
  const lowerUrl = baseUrl.toLowerCase()
  const lowerModel = model.toLowerCase()
  if (lowerUrl.includes('api.deepseek.com') || lowerModel.includes('deepseek')) return 'deepseek'
  if (lowerUrl.includes('hunyuan.cloud.tencent.com') || lowerModel.includes('hunyuan')) return 'tencent'
  if (lowerUrl.includes('dashscope.aliyuncs.com') || lowerModel.includes('qwen')) return 'aliyun'
  return 'custom'
}

function getPreset(presetKey: ProviderPresetKey): ProviderPreset | undefined {
  return PROVIDER_PRESETS.find((item) => item.key === presetKey)
}

function readLocalStorage(key: string): string {
  if (typeof window === 'undefined') return ''
  return window.localStorage.getItem(key) ?? ''
}

function isMissingSessionError(err: unknown): boolean {
  if (!(err instanceof Error)) return false
  const message = err.message.toLowerCase()
  return message.includes('session not found') || message.includes('session file missing') || message.includes('http 404')
}

function toPageCacheVersion(updatedAt: string): number {
  const parsed = Date.parse(updatedAt)
  if (!Number.isFinite(parsed) || parsed <= 0) return 0
  return parsed
}

function mergeCacheVersionFromState(prev: Record<number, number>, nextState: SessionState): Record<number, number> {
  const next = { ...prev }
  for (const page of nextState.page_states) {
    const version = toPageCacheVersion(page.updated_at)
    if (version <= 0) continue
    next[page.page_no] = Math.max(next[page.page_no] ?? 0, version)
  }
  return next
}

function toCnStatus(status: SessionState['overall_status']): string {
  const map: Record<SessionState['overall_status'], string> = {
    created: '\u5df2\u521b\u5efa',
    running: '\u8fdb\u884c\u4e2d',
    ready: '\u5df2\u5b8c\u6210',
    failed: '\u5931\u8d25',
    expired: '\u5df2\u8fc7\u671f',
    deleted: '\u5df2\u5220\u9664',
  }
  return map[status]
}

function App() {
  const [file, setFile] = useState<File | null>(null)
  const [session, setSession] = useState<SessionCreateResponse | null>(null)
  const [state, setState] = useState<SessionState | null>(null)
  const [restoringSession, setRestoringSession] = useState(true)
  const [loading, setLoading] = useState(false)
  const [starting, setStarting] = useState(false)
  const [savingPdf, setSavingPdf] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [info, setInfo] = useState<string | null>(null)
  const [cacheVersion, setCacheVersion] = useState<Record<number, number>>({})

  const [primaryId, setPrimaryId] = useState(DEFAULT_PRIMARY_ID)
  const [primaryModel, setPrimaryModel] = useState(DEFAULT_PRIMARY_MODEL)
  const [primaryBaseUrl, setPrimaryBaseUrl] = useState(DEFAULT_PRIMARY_BASE_URL)
  const [primaryKey, setPrimaryKey] = useState(() => {
    const local = readLocalStorage(PRIMARY_KEY_STORAGE).trim()
    return DEFAULT_PRIMARY_API_KEY || local
  })
  const [backupEnabled, setBackupEnabled] = useState(true)
  const [backupId, setBackupId] = useState(DEFAULT_BACKUP_ID)
  const [backupModel, setBackupModel] = useState(DEFAULT_BACKUP_MODEL)
  const [backupBaseUrl, setBackupBaseUrl] = useState(DEFAULT_BACKUP_BASE_URL)
  const [backupKey, setBackupKey] = useState(() => {
    const local = readLocalStorage(BACKUP_KEY_STORAGE).trim()
    return DEFAULT_BACKUP_API_KEY || local
  })
  const [primaryPreset, setPrimaryPreset] = useState<ProviderPresetKey>(() =>
    inferPresetKey(DEFAULT_PRIMARY_BASE_URL, DEFAULT_PRIMARY_MODEL),
  )
  const [backupPreset, setBackupPreset] = useState<ProviderPresetKey>(() =>
    inferPresetKey(DEFAULT_BACKUP_BASE_URL, DEFAULT_BACKUP_MODEL),
  )

  const leftRef = useRef<HTMLDivElement | null>(null)
  const rightRef = useRef<HTMLDivElement | null>(null)
  const syncingRef = useRef(false)
  const ensuredPagesRef = useRef<Set<number>>(new Set())

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

  const applyProviderPreset = (
    presetKey: ProviderPresetKey,
    setId: (value: string) => void,
    setModel: (value: string) => void,
    setBaseUrl: (value: string) => void,
  ) => {
    const preset = getPreset(presetKey)
    if (!preset || preset.key === 'custom') return
    setId(preset.id)
    setModel(preset.model)
    setBaseUrl(preset.baseUrl)
  }

  const refreshState = async (sessionId: string) => {
    const nextState = await getState(sessionId)
    setState(nextState)
    setCacheVersion((prev) => mergeCacheVersionFromState(prev, nextState))
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
        timeout_sec: 60,
      },
      style_profile: 'academic_conservative',
    }

    if (backupEnabled && backupKey.trim()) {
      payload.backup_provider = {
        id: backupId,
        model: backupModel,
        base_url: backupBaseUrl.trim() || undefined,
        api_key: backupKey,
        timeout_sec: 60,
      }
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
    window.localStorage.setItem(BACKUP_KEY_STORAGE, backupKey)
  }, [backupKey])

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
      void refreshState(session.session_id)
    })

    return () => {
      source.close()
    }
  }, [session?.session_id])

  useEffect(() => {
    ensuredPagesRef.current = new Set()
  }, [session?.session_id])

  useEffect(() => {
    if (!session || !state || !leftRef.current) return
    if (!['running', 'ready'].includes(state.overall_status)) return

    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (!entry.isIntersecting) continue
          const pageNo = Number((entry.target as HTMLElement).dataset.pageNo)
          if (!Number.isFinite(pageNo)) continue
          if (ensuredPagesRef.current.has(pageNo)) continue
          ensuredPagesRef.current.add(pageNo)
          void ensurePage(session.session_id, pageNo, 1).catch(() => undefined)
        }
      },
      {
        root: leftRef.current,
        threshold: 0.25,
      },
    )

    const targets = leftRef.current.querySelectorAll<HTMLElement>('[data-page-no]')
    targets.forEach((el) => observer.observe(el))

    return () => observer.disconnect()
  }, [session?.session_id, state?.overall_status, sortedPages.length])

  useEffect(() => {
    if (!session || !state || !['running', 'created'].includes(state.overall_status)) return
    const timer = window.setInterval(() => {
      void refreshState(session.session_id)
    }, 3000)
    return () => window.clearInterval(timer)
  }, [session?.session_id, state?.overall_status])

  useEffect(() => {
    if (!state || !error) return
    if (error.includes('\u5df2\u5728\u7ffb\u8bd1\u4e2d') && state.overall_status === 'running') {
      setError(null)
    }
  }, [state?.overall_status, error])

  const destroySession = async () => {
    if (!session) return
    await deleteSession(session.session_id)
    if (typeof window !== 'undefined') {
      window.localStorage.removeItem(LAST_SESSION_STORAGE)
    }
    setSession(null)
    setState(null)
    setCacheVersion({})
    setError(null)
    setInfo(null)
    ensuredPagesRef.current = new Set()
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

  const syncScroll = (from: HTMLDivElement, to: HTMLDivElement) => {
    if (syncingRef.current) return

    const fromMax = from.scrollHeight - from.clientHeight
    const toMax = to.scrollHeight - to.clientHeight
    if (fromMax <= 0 || toMax <= 0) return

    syncingRef.current = true
    const ratio = from.scrollTop / fromMax
    to.scrollTop = ratio * toMax
    window.requestAnimationFrame(() => {
      syncingRef.current = false
    })
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
        <section className="upload-stage">
          <div className="upload-panel panel">
            <h2>{'\u4e0a\u4f20\u79d1\u7814 PDF'}</h2>
            <p className="upload-desc">{'\u5355\u7bc7\u8bba\u6587\u5728\u7ebf\u53cc\u680f\u7ffb\u8bd1\uff0c\u652f\u6301\u524d 3 \u9875\u4f18\u5148\u53ef\u8bfb'}</p>
            <form onSubmit={onUpload} className="upload-form">
              <label className="file-picker">
                <input
                  type="file"
                  accept="application/pdf"
                  onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                />
                <span>{file ? file.name : '\u70b9\u51fb\u9009\u62e9 PDF \u6587\u4ef6'}</span>
              </label>
              <button type="submit" disabled={loading || !file}>
                {loading ? '\u4e0a\u4f20\u4e2d...' : '\u4e0a\u4f20 PDF \u5e76\u521b\u5efa\u4f1a\u8bdd'}
              </button>
            </form>
            <p className="upload-hint">{'\u5efa\u8bae 30MB \u4ee5\u5185\uff0c\u4f1a\u8bdd\u5173\u95ed\u540e\u81ea\u52a8\u6e05\u7406'}</p>
          </div>
        </section>
      )}

      {session && (
        <section className="panel controls-panel">
          <div className="controls-row">
            <label>
              {'主模型提供商'}
              <select
                value={primaryPreset}
                onChange={(e) => {
                  const presetKey = e.target.value as ProviderPresetKey
                  setPrimaryPreset(presetKey)
                  applyProviderPreset(presetKey, setPrimaryId, setPrimaryModel, setPrimaryBaseUrl)
                }}
              >
                {PROVIDER_PRESETS.map((preset) => (
                  <option key={`primary-${preset.key}`} value={preset.key}>
                    {preset.label}
                  </option>
                ))}
              </select>
            </label>
            <label>
              {'\u4e3b\u6a21\u578b ID'}
              <input value={primaryId} onChange={(e) => setPrimaryId(e.target.value)} />
            </label>
            <label>
              {'\u4e3b\u6a21\u578b\u540d\u79f0'}
              <input value={primaryModel} onChange={(e) => setPrimaryModel(e.target.value)} />
            </label>
            <label>
              {'\u4e3b\u6a21\u578b Base URL'}
              <input value={primaryBaseUrl} onChange={(e) => setPrimaryBaseUrl(e.target.value)} />
            </label>
            <label>
              {'\u4e3b\u6a21\u578b API Key'}
              <input type="password" value={primaryKey} onChange={(e) => setPrimaryKey(e.target.value)} />
            </label>
          </div>

          <div className="controls-row backup-toggle-row">
            <label className="checkbox-label">
              <input type="checkbox" checked={backupEnabled} onChange={(e) => setBackupEnabled(e.target.checked)} />
              {'\u542f\u7528\u5907\u7528\u6a21\u578b'}
            </label>
          </div>

          {backupEnabled && (
            <div className="controls-row">
              <label>
                {'备用模型提供商'}
                <select
                  value={backupPreset}
                  onChange={(e) => {
                    const presetKey = e.target.value as ProviderPresetKey
                    setBackupPreset(presetKey)
                    applyProviderPreset(presetKey, setBackupId, setBackupModel, setBackupBaseUrl)
                  }}
                >
                  {PROVIDER_PRESETS.map((preset) => (
                    <option key={`backup-${preset.key}`} value={preset.key}>
                      {preset.label}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                {'\u5907\u7528\u6a21\u578b ID'}
                <input value={backupId} onChange={(e) => setBackupId(e.target.value)} />
              </label>
              <label>
                {'\u5907\u7528\u6a21\u578b\u540d\u79f0'}
                <input value={backupModel} onChange={(e) => setBackupModel(e.target.value)} />
              </label>
              <label>
                {'\u5907\u7528\u6a21\u578b Base URL'}
                <input value={backupBaseUrl} onChange={(e) => setBackupBaseUrl(e.target.value)} />
              </label>
              <label>
                {'\u5907\u7528\u6a21\u578b API Key'}
                <input type="password" value={backupKey} onChange={(e) => setBackupKey(e.target.value)} />
              </label>
            </div>
          )}

          <div className="action-row">
            <button onClick={onStart} disabled={startBlocked}>
              {startLabel}
            </button>
            <button onClick={onSaveResultPdf} disabled={savingPdf}>
              {savingPdf ? '保存中...' : '保存当前结果 PDF'}
            </button>
            <button className="danger" onClick={destroySession}>
              {'\u7ed3\u675f\u4f1a\u8bdd'}
            </button>
            {state && (
              <span className="state-tag">
                {'\u72b6\u6001'}: {toCnStatus(state.overall_status)} |
                {' \u9996\u4e09\u9875\u53ef\u8bfb'}: {state.first_readable_ready ? '\u662f' : '\u5426'} |
                {' \u5df2\u5b8c\u6210'}: {progress.ready}/{progress.total} |
                {' \u5904\u7406\u4e2d'}: {progress.processing} |
                {' \u5f85\u5904\u7406'}: {progress.pending} |
                {' \u5931\u8d25'}: {progress.failed}
              </span>
            )}
          </div>
        </section>
      )}

      {error && <section className="panel error-panel">{error}</section>}
      {info && <section className="panel info-panel">{info}</section>}

      {session && state && (
        <main className="viewer-grid">
          <section
            className="viewer-col"
            ref={leftRef}
            onScroll={() => {
              if (leftRef.current && rightRef.current) syncScroll(leftRef.current, rightRef.current)
            }}
          >
            <h2>{'\u539f\u6587'}</h2>
            <div className="pages-stack">
              {sortedPages.map((page) => (
                <article key={`original-${page.page_no}`} className="page-card" data-page-no={page.page_no}>
                  <div className="page-meta">{'\u7b2c'} {page.page_no} {'\u9875'}</div>
                  <img loading="lazy" src={originalPageUrl(session.session_id, page.page_no)} alt={`original-${page.page_no}`} />
                </article>
              ))}
            </div>
          </section>

          <section
            className="viewer-col"
            ref={rightRef}
            onScroll={() => {
              if (leftRef.current && rightRef.current) syncScroll(rightRef.current, leftRef.current)
            }}
          >
            <h2>{'\u8bd1\u6587'}</h2>
            <div className="pages-stack">
              {sortedPages.map((page) => (
                <article key={`translated-${page.page_no}`} className="page-card">
                  <div className="page-meta">{'\u7b2c'} {page.page_no} {'\u9875'}</div>
                  {page.status === 'ready' && (
                    <>
                      <img
                        loading="lazy"
                        src={translatedPageUrl(session.session_id, page.page_no, cacheVersion[page.page_no] ?? 0)}
                        alt={`translated-${page.page_no}`}
                      />
                      <button onClick={() => onRetryPage(page.page_no)}>{'重译该页'}</button>
                    </>
                  )}
                  {page.status === 'pending' && <div className="placeholder">{'\u7b49\u5f85\u8fdb\u5165\u7ffb\u8bd1\u961f\u5217...'}</div>}
                  {page.status === 'processing' && <div className="placeholder">{'\u540e\u53f0\u7ffb\u8bd1\u4e2d...'}</div>}
                  {page.status === 'failed' && (
                    <div className="placeholder failed">
                      <div>{'\u7ffb\u8bd1\u5931\u8d25'}: {page.error ?? '\u672a\u77e5\u9519\u8bef'}</div>
                      <button onClick={() => onRetryPage(page.page_no)}>
                        {'\u91cd\u8bd5\u8be5\u9875'}
                      </button>
                    </div>
                  )}
                </article>
              ))}
            </div>
          </section>

        </main>
      )}
    </div>
  )
}

export default App
