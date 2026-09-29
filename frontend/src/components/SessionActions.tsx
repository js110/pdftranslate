import type { SessionState } from '../types'
import { toCnStatus } from '../constants'

type Props = {
  state: SessionState | null
  progress: { total: number; ready: number; processing: number; pending: number; failed: number }
  startLabel: string
  startBlocked: boolean
  savingPdf: boolean
  exportingPdf: boolean
  onStart: () => void
  onSaveResultPdf: () => void
  onExportResultPdf: () => void
  onDestroySession: () => void
}

export function SessionActions(props: Props) {
  const {
    state,
    progress,
    startLabel,
    startBlocked,
    savingPdf,
    exportingPdf,
    onStart,
    onSaveResultPdf,
    onExportResultPdf,
    onDestroySession,
  } = props

  return (
    <>
      <div className="action-row">
        <button onClick={onStart} disabled={startBlocked}>
          {startLabel}
        </button>
        <button onClick={onSaveResultPdf} disabled={savingPdf}>
          {savingPdf ? '保存中...' : '保存当前结果 PDF'}
        </button>
        <button onClick={onExportResultPdf} disabled={exportingPdf}>
          {exportingPdf ? '\u5bfc\u51fa\u4e2d...' : '\u5bfc\u51fa\u7ffb\u8bd1PDF'}
        </button>
        <button className="danger" onClick={onDestroySession}>
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

      {state?.overall_status === 'running' && progress.total > 0 && (
        <div className="progress-bar-wrap">
          <div
            className="progress-bar"
            style={{ width: `${Math.round((progress.ready / progress.total) * 100)}%` }}
          />
          <span className="progress-label">
            {Math.round((progress.ready / progress.total) * 100)}% ({progress.ready}/{progress.total})
          </span>
        </div>
      )}
    </>
  )
}
