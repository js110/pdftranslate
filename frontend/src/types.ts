export type PageStatus = 'pending' | 'processing' | 'ready' | 'failed'

export interface PageState {
  page_no: number
  status: PageStatus
  warnings: string[]
  error: string | null
  updated_at: string
}

export interface SessionCreateResponse {
  session_id: string
  page_count: number
  expires_at: string
}

export interface ProviderConfigInput {
  id: string
  model: string
  api_key: string
  base_url?: string
  timeout_sec: number
}

export interface StartSessionRequest {
  primary_provider: ProviderConfigInput
  backup_provider?: ProviderConfigInput
  style_profile: string
}

export interface EnsurePageRequest {
  window: number
}

export interface EnsurePageResponse {
  session_id: string
  requested_page: number
  window: number
  queued_pages: number[]
}

export interface SaveResultPdfResponse {
  session_id: string
  saved_path: string
  page_count: number
  translated_pages: number
}

export interface SessionState {
  session_id: string
  overall_status: 'created' | 'running' | 'ready' | 'failed' | 'expired' | 'deleted'
  page_count: number
  page_states: PageState[]
  first_readable_ready: boolean
  warnings_count: number
  created_at: string
  expires_at: string
}

export interface QualityReport {
  layout_overflow: Array<{ page_no: number; bbox: number[]; reason: string }>
  image_ocr_failures: Array<{ page_no: number; image_index: number; reason: string }>
  fallback_events: Array<{ page_no: number; from_provider: string; to_provider: string; reason: string }>
  untranslated_blocks: Array<{ page_no: number; bbox: number[]; source_excerpt: string; reason: string }>
}

export interface EventEnvelope {
  event:
    | 'page_ready'
    | 'page_failed'
    | 'provider_switched'
    | 'session_expiring'
    | 'session_ready'
    | 'session_failed'
    | 'progress'
  session_id: string
  payload: Record<string, unknown>
  ts: string
}
