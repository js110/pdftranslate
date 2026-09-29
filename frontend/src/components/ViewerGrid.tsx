import { useEffect, useMemo, useRef, useState } from 'react'
import { ensurePage, fetchPageBlocks, originalPageUrl, translatedPageUrl } from '../api'
import type { PageBlocks, PageState, SessionCreateResponse, SessionState } from '../types'
import { SHOW_RETRANSLATE_BUTTON } from '../constants'

type Props = {
  session: SessionCreateResponse
  state: SessionState
  cacheVersion: Record<number, number>
  onRetryPage: (pageNo: number) => void
}

type ViewMode = 'reflow' | 'image'

const VIEW_MODE_STORAGE = 'pdftranslate_view_mode'
const REFLOW_BASE_PX = 15.5
const REFLOW_MIN_PX = 12.5
const REFLOW_MAX_PX = 27
const HEADING_RATIO = 1.18
const FOOTNOTE_RATIO = 0.86

function syncScroll(from: HTMLDivElement, to: HTMLDivElement, syncingRef: { current: boolean }) {
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

function medianTextFontSize(blocks: PageBlocks['blocks']): number {
  const sizes = blocks
    .filter((block) => block.kind === 'text' && block.font_size > 0)
    .map((block) => block.font_size)
    .sort((a, b) => a - b)
  if (!sizes.length) return 0
  return sizes[Math.floor(sizes.length / 2)]
}

function ReflowPage({ blocks }: { blocks: PageBlocks }) {
  const median = useMemo(() => medianTextFontSize(blocks.blocks), [blocks])

  return (
    <div className="reflow-page">
      {blocks.blocks.map((block, index) => {
        const ratio = median > 0 ? block.font_size / median : 1
        const className =
          ratio >= HEADING_RATIO ? 'reflow-block heading' : ratio <= FOOTNOTE_RATIO ? 'reflow-block footnote' : 'reflow-block'
        const px = Math.min(REFLOW_MAX_PX, Math.max(REFLOW_MIN_PX, REFLOW_BASE_PX * ratio))
        return (
          <p key={`text-${index}`} className={className} style={{ fontSize: `${px.toFixed(1)}px` }}>
            {block.translated_text}
          </p>
        )
      })}
    </div>
  )
}

function TranslatedPageCard({
  session,
  page,
  version,
  viewMode,
  onRetryPage,
}: {
  session: SessionCreateResponse
  page: PageState
  version: number
  viewMode: ViewMode
  onRetryPage: (pageNo: number) => void
}) {
  const [blocks, setBlocks] = useState<PageBlocks | null>(null)

  useEffect(() => {
    if (viewMode !== 'reflow' || page.status !== 'ready') return
    let cancelled = false
    setBlocks(null)
    void fetchPageBlocks(session.session_id, page.page_no, version).then((data) => {
      if (!cancelled) setBlocks(data)
    })
    return () => {
      cancelled = true
    }
  }, [session.session_id, page.page_no, page.status, version, viewMode])

  return (
    <article className="page-card">
      <div className="page-meta">第 {page.page_no} 页</div>
      {page.status === 'ready' && (
        <>
          {viewMode === 'reflow' && blocks && blocks.blocks.length > 0 ? (
            <ReflowPage blocks={blocks} />
          ) : (
            <img
              loading="lazy"
              src={translatedPageUrl(session.session_id, page.page_no, version)}
              alt={`translated-${page.page_no}`}
            />
          )}
          {SHOW_RETRANSLATE_BUTTON && <button onClick={() => onRetryPage(page.page_no)}>重译该页</button>}
        </>
      )}
      {page.status === 'pending' && <div className="placeholder">等待进入翻译队列...</div>}
      {page.status === 'processing' && <div className="placeholder">后台翻译中...</div>}
      {page.status === 'failed' && (
        <div className="placeholder failed">
          <div>翻译失败: {page.error ?? '未知错误'}</div>
          <button onClick={() => onRetryPage(page.page_no)}>重试该页</button>
        </div>
      )}
    </article>
  )
}

export function ViewerGrid({ session, state, cacheVersion, onRetryPage }: Props) {
  const leftRef = useRef<HTMLDivElement | null>(null)
  const rightRef = useRef<HTMLDivElement | null>(null)
  const syncingRef = useRef(false)
  const ensuredPagesRef = useRef<Set<number>>(new Set())
  const [viewMode, setViewMode] = useState<ViewMode>(() => {
    if (typeof window === 'undefined') return 'reflow'
    return window.localStorage.getItem(VIEW_MODE_STORAGE) === 'image' ? 'image' : 'reflow'
  })

  const switchViewMode = (mode: ViewMode) => {
    setViewMode(mode)
    try {
      window.localStorage.setItem(VIEW_MODE_STORAGE, mode)
    } catch {
      // localStorage unavailable: keep in-memory mode only
    }
  }

  useEffect(() => {
    ensuredPagesRef.current = new Set()
  }, [session?.session_id])

  const sortedPages = useMemo(() => [...state.page_states].sort((a, b) => a.page_no - b.page_no), [state.page_states])

  useEffect(() => {
    if (!leftRef.current) return
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
  }, [session?.session_id, state.overall_status, sortedPages.length])

  return (
    <main className="viewer-grid">
      <section
        className="viewer-col"
        ref={leftRef}
        onScroll={() => {
          if (leftRef.current && rightRef.current) syncScroll(leftRef.current, rightRef.current, syncingRef)
        }}
      >
        <h2>原文</h2>
        <div className="pages-stack">
          {sortedPages.map((page) => (
            <article key={`original-${page.page_no}`} className="page-card" data-page-no={page.page_no}>
              <div className="page-meta">第 {page.page_no} 页</div>
              <img loading="lazy" src={originalPageUrl(session.session_id, page.page_no)} alt={`original-${page.page_no}`} />
            </article>
          ))}
        </div>
      </section>

      <section
        className="viewer-col"
        ref={rightRef}
        onScroll={() => {
          if (leftRef.current && rightRef.current) syncScroll(rightRef.current, leftRef.current, syncingRef)
        }}
      >
        <h2>
          译文
          <span className="view-toggle">
            <button className={viewMode === 'reflow' ? 'active' : ''} onClick={() => switchViewMode('reflow')}>
              排版阅读
            </button>
            <button className={viewMode === 'image' ? 'active' : ''} onClick={() => switchViewMode('image')}>
              原版式
            </button>
          </span>
        </h2>
        <div className="pages-stack">
          {sortedPages.map((page) => (
            <TranslatedPageCard
              key={`translated-${page.page_no}`}
              session={session}
              page={page}
              version={cacheVersion[page.page_no] ?? 0}
              viewMode={viewMode}
              onRetryPage={onRetryPage}
            />
          ))}
        </div>
      </section>
    </main>
  )
}
