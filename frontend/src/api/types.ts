export interface User {
  id: number
  username: string
  created_at?: string
  last_login_at?: string
}

export interface TokenOut {
  access_token: string
  token_type: string
  expires_in: number
  user: User
}

export interface CaptchaOut {
  captcha_id: string
  image_base64: string
  expires_in: number
}

export interface Kb {
  id: string
  name: string
  description: string
  doc_count: number
  chunk_count: number
  created_at?: string
  updated_at?: string
}

export interface KbStats {
  kb_id: string
  documents: number
  chunks: number
  enabled_chunks: number
  failed_documents: number
}

export type DocStatus =
  | 'uploaded'
  | 'parsing'
  | 'chunking'
  | 'embedding'
  | 'done'
  | 'failed'

export interface Doc {
  id: string
  kb_id: string
  filename: string
  file_type: string
  size_bytes: number
  page_count: number
  status: DocStatus
  last_stage: string
  error_msg: string
  parse_params: Record<string, unknown>
  created_at?: string
  duplicated: boolean
}

export interface Chunk {
  id: string
  doc_id: string
  kb_id: string
  parent_id?: string | null
  seq: number
  content: string
  char_count: number
  page_start: number
  page_end: number
  enabled: boolean
  meta: Record<string, any>
}

export interface ChunkPage {
  items: Chunk[]
  total: number
  page: number
  size: number
}

export interface Citation {
  sid: string
  doc: string
  page: number | null
  chunk_id: string
  excerpt: string
  doc_id: string
}

export interface Session {
  id: string
  title: string
  kb_ids: string[]
  model: string
  message_count: number
  created_at?: string
  updated_at?: string
}

export interface ChatMessage {
  id: number
  session_id: string
  role: 'user' | 'assistant'
  content: string
  citations: Citation[]
  tokens: number
  latency_ms: number
  created_at?: string
}

export interface MessagePage {
  items: ChatMessage[]
  total: number
  page: number
  size: number
}

/** 后端统一错误体：{ code, message, trace_id, details? } */
export interface ApiErrorBody {
  code: string
  message: string
  trace_id?: string
  details?: Array<{ field: string; reason: string }>
}

export interface RecommendItem {
  chunk_id: string
  title: string
  preview: string
  doc_id: string
  doc_title: string
  page: number
  char_count: number
}

export interface RecommendList {
  date: string
  variation: number
  total_candidates: number
  items: RecommendItem[]
}

export interface RelatedItem {
  chunk_id: string
  doc_id: string
  doc_title: string
  page: number
  preview: string
}

export interface RecommendDetail extends RecommendItem {
  content: string
  is_full_parent: boolean
  related: RelatedItem[]
}
