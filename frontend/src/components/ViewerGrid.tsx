import { useEffect, useRef } from 'react'
import { ensurePage } from '../api'
import { originalPageUrl, translatedPageUrl } from '../api'
import type { SessionCreateResponse, SessionState } from '../types'
import { SHOW_RETRANSLATE_BUTTON } from '../constants'

type Props = {
  session: SessionCreateResponse
  state: SessionState
  cacheVersion: Record<number, number>
  onRetryPage: (pageNo: number) => void
}

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

export function ViewerGrid({ session, state, cacheVersion, onRetryPage }: Props) {
  const leftRef = useRef<HTMLDivElement | null>(null)
  const rightRef = useRef<HTMLDivElement | null>(null)
  const syncingRef = useRef(false)
  const ensuredPagesRef = useRef<Set<number>>(new Set())

  useEffect(() => {
    ensuredPagesRef.current = new Set()
  }, [session?.session_id])

  const sortedPages = [...state.page_states].sort((a, b) => a.page_no - b.page_no)

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
          if (leftRef.current && rightRef.current) syncScroll(rightRef.current, leftRef.current, syncingRef)
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
                  {SHOW_RETRANSLATE_BUTTON && <button onClick={() => onRetryPage(page.page_no)}>{'重译该页'}</button>}
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
  )
}
