import { auth } from './client'
import type { ApiErrorBody, Citation } from './types'

export interface MetaPayload {
  session_id: string
  is_new: boolean
}

export interface SourcesPayload {
  citations: Citation[]
  hit_count: number
  intent: string
  strategy: string
  retrieve_ms: number
}

export interface DonePayload {
  no_context: boolean
  text_len: number
  dangling: string[]
  coverage: number
  tokens_in: number
  tokens_out: number
  latency_ms: number
  message_id: number | null
  safety?: string
}

export interface StreamErrorPayload {
  code: string
  message: string
}

export interface StreamHandlers {
  onMeta?: (data: MetaPayload) => void
  onSources?: (data: SourcesPayload) => void
  onToken?: (text: string) => void
  onDone?: (data: DonePayload) => void
  onError?: (data: StreamErrorPayload) => void
}

export interface ChatRequestBody {
  question: string
  kb_ids: string[]
  session_id?: string | null
  user_level?: string
}

/**
 * 流式问答。
 *
 * 用 fetch 而不是 EventSource：EventSource 不支持自定义 Authorization 头，
 * 而 Cookie 方案我们没用（token 在 localStorage）。也别用 axios——
 * 它基于 XHR，拿不到真正的分块。
 */
export async function streamChat(
  body: ChatRequestBody,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${auth.token}`,
    },
    body: JSON.stringify(body),
    signal,
  })

  if (!response.ok) {
    let message = `请求失败（HTTP ${response.status}）`
    try {
      const payload = (await response.json()) as ApiErrorBody
      if (payload?.message) message = payload.message
    } catch {
      // 后端返回的不是 JSON（比如代理的 502 HTML），用默认文案
    }
    throw new Error(message)
  }
  if (!response.body) {
    throw new Error('当前浏览器不支持流式响应')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    let cut = buffer.indexOf('\n\n')
    while (cut !== -1) {
      dispatch(buffer.slice(0, cut), handlers)
      buffer = buffer.slice(cut + 2)
      cut = buffer.indexOf('\n\n')
    }
  }

  // 少数代理不会补最后一个空行，残留自己也解析一次
  if (buffer.trim()) dispatch(buffer, handlers)
}

function dispatch(block: string, handlers: StreamHandlers): void {
  let event = ''
  const dataLines: string[] = []

  for (const line of block.split(/\r?\n/)) {
    if (!line || line.startsWith(':')) continue
    if (line.startsWith('event:')) event = line.slice(6).trim()
    else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim())
  }
  if (!event) return

  const raw = dataLines.join('\n')
  let data: any = {}
  if (raw) {
    try {
      data = JSON.parse(raw)
    } catch {
      return
    }
  }

  switch (event) {
    case 'meta':
      handlers.onMeta?.(data)
      break
    case 'sources':
      handlers.onSources?.(data)
      break
    case 'token':
      handlers.onToken?.(String(data.text ?? ''))
      break
    case 'done':
      handlers.onDone?.(data)
      break
    case 'error':
      handlers.onError?.(data)
      break
    default:
      break
  }
}
