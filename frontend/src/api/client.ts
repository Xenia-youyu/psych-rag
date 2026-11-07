import axios, { AxiosError } from 'axios'

import type { ApiErrorBody } from './types'

const TOKEN_KEY = 'psy_token'
const USER_KEY = 'psy_user'

export const auth = {
  get token(): string {
    return localStorage.getItem(TOKEN_KEY) || ''
  },
  get user() {
    const raw = localStorage.getItem(USER_KEY)
    if (!raw) return null
    try {
      return JSON.parse(raw)
    } catch {
      return null
    }
  },
  save(token: string, user: unknown) {
    localStorage.setItem(TOKEN_KEY, token)
    localStorage.setItem(USER_KEY, JSON.stringify(user))
  },
  clear() {
    localStorage.removeItem(TOKEN_KEY)
    localStorage.removeItem(USER_KEY)
  },
}

/** 把后端错误体掏成一句能显示给人看的话。 */
export function errorMessage(error: unknown): string {
  const axiosError = error as AxiosError<ApiErrorBody>
  const body = axiosError?.response?.data
  if (body?.message) {
    // 参数校验失败时 message 只有一句笼统的「请求参数不合法」，
    // 真正有用的原因在 details 里（例如「密码长度需为 8~64 位」）。
    // 只显示 message 等于把唯一的诊断信息扔掉，用户只能猜。
    const reasons = (body.details ?? [])
      .map((d) => d.reason)
      // Pydantic 的 msg 形如 "Value error, 密码需同时包含字母与数字"，前面是噪声
      .map((r) => r.replace(/^Value error,\s*/, ''))
      .filter((r) => r && r !== 'Field required')
    if (reasons.length) return reasons.join('；')
    return body.message
  }
  if (axiosError?.code === 'ECONNABORTED') {
    return '请求超时，请稍后重试'
  }
  if (axiosError?.message === 'Network Error') {
    return '连不上后端服务（确认 uvicorn 已在 8000 端口启动）'
  }
  return (error as Error)?.message || '请求失败'
}

export const http = axios.create({
  baseURL: '',
  timeout: 30000,
})

http.interceptors.request.use((config) => {
  if (auth.token) {
    config.headers.Authorization = `Bearer ${auth.token}`
  }
  return config
})

let redirecting = false

http.interceptors.response.use(
  (response) => response,
  (error: AxiosError<ApiErrorBody>) => {
    const status = error.response?.status
    const code = error.response?.data?.code

    // token 过期/无效才踢回登录页。403/404 是业务语义，不该把人踢出去。
    if (status === 401 && !redirecting) {
      redirecting = true
      auth.clear()
      const current = window.location.hash || '#/kb'
      if (!current.startsWith('#/login')) {
        window.location.hash = `#/login?redirect=${encodeURIComponent(current)}`
      }
      setTimeout(() => (redirecting = false), 1000)
    }
    if (status === 401 && (code === 'TOKEN_EXPIRED' || code === 'TOKEN_INVALID')) {
      auth.clear()
    }
    return Promise.reject(error)
  },
)
