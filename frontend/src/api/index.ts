import { http } from './client'
import type {
  CaptchaOut,
  Chunk,
  ChunkPage,
  Doc,
  Kb,
  KbStats,
  MessagePage,
  RecommendDetail,
  RecommendList,
  Session,
  TokenOut,
  User,
} from './types'

export const api = {
  // ---------------------------------------------------------------- 鉴权
  // scene 必须与提交时的场景一致：服务端会校验 record.scene，
  // 拿登录的码去注册会报「验证码与场景不匹配」
  captcha: (scene: 'login' | 'register' = 'login') =>
    http.get<CaptchaOut>('/api/auth/captcha', { params: { scene } }).then((r) => r.data),
  register: (payload: {
    username: string
    password: string
    password_confirm: string
    captcha_id: string
    captcha_code: string
  }) => http.post('/api/auth/register', payload).then((r) => r.data),
  login: (payload: { username: string; password: string; captcha_id: string; captcha_code: string }) =>
    http.post<TokenOut>('/api/auth/login', payload).then((r) => r.data),
  me: () => http.get<User>('/api/auth/me').then((r) => r.data),

  // ---------------------------------------------------------------- 知识库
  listKbs: () => http.get<Kb[]>('/api/kb').then((r) => r.data),
  createKb: (payload: { name: string; description: string }) =>
    http.post<Kb>('/api/kb', payload).then((r) => r.data),
  getKb: (id: string) => http.get<Kb>(`/api/kb/${id}`).then((r) => r.data),
  updateKb: (id: string, payload: { name?: string; description?: string }) =>
    http.patch<Kb>(`/api/kb/${id}`, payload).then((r) => r.data),
  deleteKb: (id: string) => http.delete(`/api/kb/${id}`).then((r) => r.data),
  kbStats: (id: string) => http.get<KbStats>(`/api/kb/${id}/stats`).then((r) => r.data),

  // ---------------------------------------------------------------- 文档
  listDocs: (kbId: string) => http.get<Doc[]>(`/api/kb/${kbId}/documents`).then((r) => r.data),
  getDoc: (id: string) => http.get<Doc>(`/api/documents/${id}`).then((r) => r.data),
  deleteDoc: (id: string) => http.delete(`/api/documents/${id}`).then((r) => r.data),
  supportedTypes: () =>
    http.get<{ extensions: string[]; max_size_mb: number }>('/api/supported-types').then((r) => r.data),

  uploadDoc: (
    kbId: string,
    file: File,
    params: Record<string, string | number | boolean>,
    onProgress?: (percent: number) => void,
  ) => {
    const body = new FormData()
    body.append('file', file)
    Object.entries(params).forEach(([key, value]) => body.append(key, String(value)))
    return http
      .post<Doc>(`/api/kb/${kbId}/documents`, body, {
        // 一本 500 页的书光上传就可能几十秒，别用默认超时
        timeout: 0,
        onUploadProgress: (event) => {
          if (onProgress && event.total) {
            onProgress(Math.round((event.loaded / event.total) * 100))
          }
        },
      })
      .then((r) => r.data)
  },

  // ---------------------------------------------------------------- 文本块
  listChunks: (
    docId: string,
    query: { page?: number; size?: number; keyword?: string; only_enabled?: boolean } = {},
  ) =>
    http
      .get<ChunkPage>(`/api/documents/${docId}/chunks`, { params: query })
      .then((r) => r.data),
  updateChunk: (chunkId: string, payload: { content?: string; enabled?: boolean }) =>
    http.patch<Chunk>(`/api/chunks/${chunkId}`, payload).then((r) => r.data),
  deleteChunk: (chunkId: string) => http.delete(`/api/chunks/${chunkId}`).then((r) => r.data),
  pendingReindex: (docId: string) =>
    http.get<{ doc_id: string; pending: number }>(`/api/documents/${docId}/pending-reindex`).then((r) => r.data),

  // ---------------------------------------------------------------- 每日推荐
  recommendDaily: (kbId: string, variation = 0, limit = 3) =>
    http
      .get<RecommendList>('/api/recommend/daily', {
        params: { kb_id: kbId, variation, limit },
      })
      .then((r) => r.data),
  recommendDetail: (chunkId: string) =>
    http.get<RecommendDetail>(`/api/recommend/chunks/${chunkId}`).then((r) => r.data),

  // ---------------------------------------------------------------- 会话
  listSessions: () => http.get<Session[]>('/api/sessions').then((r) => r.data),
  getSession: (id: string) => http.get<Session>(`/api/sessions/${id}`).then((r) => r.data),
  renameSession: (id: string, title: string) =>
    http.patch<Session>(`/api/sessions/${id}`, { title }).then((r) => r.data),
  deleteSession: (id: string) => http.delete(`/api/sessions/${id}`).then((r) => r.data),
  listMessages: (id: string) =>
    http.get<MessagePage>(`/api/sessions/${id}/messages`, { params: { size: 200 } }).then((r) => r.data),
}
