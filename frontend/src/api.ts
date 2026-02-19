import type {
  EnsurePageResponse,
  EventEnvelope,
  SaveResultPdfResponse,
  SessionCreateResponse,
  SessionState,
  StartSessionRequest,
} from './types'

const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? 'http://localhost:8000/v1'

function toErrorMessage(data: unknown): string {
  if (typeof data === 'object' && data && 'detail' in data) {
    return String((data as { detail: unknown }).detail)
  }
  return 'Request failed'
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(`${API_BASE}${path}`, init)
  if (!resp.ok) {
    let message = `HTTP ${resp.status}`
    try {
      const body = await resp.json()
      message = toErrorMessage(body)
    } catch {
      // no-op
    }
    throw new Error(message)
  }
  if (resp.status === 204) {
    return undefined as T
  }
  return (await resp.json()) as T
}

export async function createSession(file: File): Promise<SessionCreateResponse> {
  const formData = new FormData()
  formData.append('file', file)
  return request<SessionCreateResponse>('/sessions', {
    method: 'POST',
    body: formData,
  })
}

export async function startSession(sessionId: string, payload: StartSessionRequest): Promise<void> {
  await request(`/sessions/${sessionId}/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
}

export async function getState(sessionId: string): Promise<SessionState> {
  return request<SessionState>(`/sessions/${sessionId}/state`)
}

export async function retryPage(sessionId: string, pageNo: number): Promise<void> {
  await request(`/sessions/${sessionId}/pages/${pageNo}/retry`, { method: 'POST' })
}

export async function ensurePage(sessionId: string, pageNo: number, window = 1): Promise<EnsurePageResponse> {
  return request<EnsurePageResponse>(`/sessions/${sessionId}/pages/${pageNo}/ensure`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ window }),
  })
}

export async function saveResultPdf(sessionId: string): Promise<SaveResultPdfResponse> {
  return request<SaveResultPdfResponse>(`/sessions/${sessionId}/save-result-pdf`, {
    method: 'POST',
  })
}

export async function deleteSession(sessionId: string): Promise<void> {
  await request(`/sessions/${sessionId}`, { method: 'DELETE' })
}

export function originalPageUrl(sessionId: string, pageNo: number): string {
  return `${API_BASE}/sessions/${sessionId}/pages/${pageNo}/original.png`
}

export function translatedPageUrl(sessionId: string, pageNo: number, version = 0): string {
  return `${API_BASE}/sessions/${sessionId}/pages/${pageNo}/translated.png?v=${version}`
}

export function subscribeEvents(sessionId: string, onEvent: (event: EventEnvelope) => void): EventSource {
  const source = new EventSource(`${API_BASE}/sessions/${sessionId}/events`)
  source.onmessage = (msg) => {
    try {
      const parsed = JSON.parse(msg.data) as EventEnvelope
      onEvent(parsed)
    } catch {
      // no-op
    }
  }
  return source
}
