import { defineStore } from 'pinia'
import { computed, ref } from 'vue'

import { api } from '@/api'
import { auth } from '@/api/client'
import type { User } from '@/api/types'

export const useAuthStore = defineStore('auth', () => {
  const user = ref<User | null>(auth.user as User | null)
  const token = ref<string>(auth.token)
  const isLoggedIn = computed(() => Boolean(token.value))

  async function login(payload: {
    username: string
    password: string
    captcha_id: string
    captcha_code: string
  }) {
    const data = await api.login(payload)
    auth.save(data.access_token, data.user)
    token.value = data.access_token
    user.value = data.user
    return data.user
  }

  /** 刷新页后校验 token 是否还有效，顺带拿到最新用户名。 */
  async function refresh() {
    if (!token.value) return
    try {
      user.value = await api.me()
      auth.save(token.value, user.value)
    } catch {
      logout()
    }
  }

  function logout() {
    auth.clear()
    token.value = ''
    user.value = null
  }

  return { user, token, isLoggedIn, login, refresh, logout }
})
