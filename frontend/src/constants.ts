import type { SessionState } from './types'

export const PRIMARY_KEY_STORAGE = 'pdftranslate_primary_api_key'
export const LAST_SESSION_STORAGE = 'pdftranslate_last_session_id'

export const DEFAULT_PRIMARY_ID = (import.meta.env.VITE_DEFAULT_PRIMARY_ID as string | undefined) ?? 'deepseek-main'
export const DEFAULT_PRIMARY_MODEL = (import.meta.env.VITE_DEFAULT_PRIMARY_MODEL as string | undefined) ?? 'deepseek-chat'
export const DEFAULT_PRIMARY_BASE_URL =
  (import.meta.env.VITE_DEFAULT_PRIMARY_BASE_URL as string | undefined) ?? 'https://api.deepseek.com/v1'
export const DEFAULT_PRIMARY_API_KEY = (import.meta.env.VITE_DEFAULT_PRIMARY_API_KEY as string | undefined) ?? ''
export const SHOW_RETRANSLATE_BUTTON =
  import.meta.env.DEV || ((import.meta.env.VITE_ENABLE_RETRANSLATE_BUTTON as string | undefined) ?? '') === 'true'

export function isMissingSessionError(err: unknown): boolean {
  if (!(err instanceof Error)) return false
  const message = err.message.toLowerCase()
  return message.includes('session not found') || message.includes('session file missing') || message.includes('http 404')
}

export function toPageCacheVersion(updatedAt: string): number {
  const parsed = Date.parse(updatedAt)
  if (!Number.isFinite(parsed) || parsed <= 0) return 0
  return parsed
}

export function mergeCacheVersionFromState(prev: Record<number, number>, nextState: SessionState): Record<number, number> {
  const next = { ...prev }
  for (const page of nextState.page_states) {
    const version = toPageCacheVersion(page.updated_at)
    if (version <= 0) continue
    next[page.page_no] = Math.max(next[page.page_no] ?? 0, version)
  }
  return next
}

export function toCnStatus(status: SessionState['overall_status']): string {
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
