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
  if (axiosError?.response?.data?.message) {
    return axiosError.response.data.message
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
