import { useEffect, useMemo, useRef, useState } from 'react'
import type { CSSProperties } from 'react'
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
const HEADING_RATIO = 1.18
const FOOTNOTE_RATIO = 0.86
// Matches .reflow-page font-family so canvas measurement agrees with layout.
const REFLOW_FONT = "Georgia, 'Times New Roman', 'Songti SC', 'SimSun', serif"

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

/** Position/size in % of the page box. */
const pct = (value: number, total: number) => `${((value / total) * 100).toFixed(3)}%`
/** Length in container-query width units (1cqw = 1% of page width). */
const cqw = (value: number, total: number) => `${((value / total) * 100).toFixed(3)}cqw`

let measureCtx: CanvasRenderingContext2D | null | undefined
const measureCache = new Map<string, number>()

function measureLine(text: string, fontSize: number, weight: number): number {
  const key = `${weight}|${fontSize.toFixed(2)}|${text}`
  const cached = measureCache.get(key)
  if (cached !== undefined) return cached
  if (measureCtx === undefined) {
    try {
      measureCtx = document.createElement('canvas').getContext('2d')
    } catch {
      measureCtx = null
    }
  }
  let width: number
  if (measureCtx) {
    measureCtx.font = `${weight} ${fontSize}px ${REFLOW_FONT}`
    width = measureCtx.measureText(text).width
  } else {
    width = text.length * fontSize * 0.6
  }
  if (measureCache.size > 4000) measureCache.clear()
  measureCache.set(key, width)
  return width
}

/**
 * Renders translated text at the original PDF block coordinates so the layout
 * mirrors the source page (columns, margins, headings, footnotes).
 */
function ReflowPage({ blocks }: { blocks: PageBlocks }) {
  const median = useMemo(() => medianTextFontSize(blocks.blocks), [blocks])
  const pageWidth = blocks.width > 0 ? blocks.width : 1
  const pageHeight = blocks.height > 0 ? blocks.height : 1

  const laidOut = useMemo(() => {
    const maxWidth = blocks.blocks.reduce((max, block) => Math.max(max, block.bbox[2] - block.bbox[0]), 0)
    return blocks.blocks.map((block, index) => {
      const [x0, y0, x1, y1] = block.bbox
      const width = Math.max(1, x1 - x0)
      const height = Math.max(1, y1 - y0)
      const baseFont = block.font_size > 0 ? block.font_size : median || 33
      const ratio = median > 0 ? baseFont / median : 1
      const weight: 400 | 700 = ratio >= HEADING_RATIO ? 700 : 400
      const outLines = block.translated_text.split('\n')
      const widest = outLines.reduce((max, line) => Math.max(max, measureLine(line, baseFont, weight)), 0)
      // Shrink to fit the original line box; past a point let it wrap instead.
      const fit = width > 0 && widest > 0 ? (width * 0.97) / widest : 1
      const scale = Math.min(1, Math.max(0.72, fit))
      const fontPx = baseFont * scale
      // How many rendered lines this text occupies after wrapping.
      const wrapWidth = width * 0.97
      const totalLines = outLines.reduce((sum, line) => {
        const w = measureLine(line, baseFont, weight) * scale
        return sum + (wrapWidth > 0 ? Math.max(1, Math.ceil(w / wrapWidth)) : 1)
      }, 0)
      // Single-line text is vertically centered in its original line box;
      // multi-line keeps the source line spacing but never below ~1.18em.
      const lineHeightPx = Math.max(height / Math.max(1, totalLines), fontPx * 1.18)
      const centered =
        ratio >= HEADING_RATIO ||
        (width < maxWidth * 0.85 && Math.abs((x0 + x1) / 2 - pageWidth / 2) < pageWidth * 0.05)
      const className =
        'reflow-block' +
        (ratio >= HEADING_RATIO ? ' heading' : '') +
        (ratio <= FOOTNOTE_RATIO ? ' footnote' : '')
      const style: CSSProperties = {
        left: pct(x0, pageWidth),
        top: pct(y0, pageHeight),
        width: pct(width, pageWidth),
        height: pct(height, pageHeight),
        fontSize: cqw(fontPx, pageWidth),
        lineHeight: cqw(lineHeightPx, pageWidth),
        fontWeight: weight,
        whiteSpace: 'pre-wrap',
        textAlign: centered ? 'center' : fit < 1 ? 'justify' : 'left',
      }
      return { key: `b-${index}`, text: block.translated_text, className, style }
    })
  }, [blocks, median, pageWidth, pageHeight])

  return (
    <div className="reflow-page" style={{ aspectRatio: `${pageWidth} / ${pageHeight}` }}>
      {laidOut.map((item) => (
        <p key={item.key} className={item.className} style={item.style}>
          {item.text}
        </p>
      ))}
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
