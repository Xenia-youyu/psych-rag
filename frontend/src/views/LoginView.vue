<script setup lang="ts">
import { Lock, Refresh, User } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import { onMounted, reactive, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'

import { api } from '@/api'
import { errorMessage } from '@/api/client'
import type { CaptchaOut } from '@/api/types'
import { useAuthStore } from '@/stores/auth'

const route = useRoute()
const router = useRouter()
const authStore = useAuthStore()

const mode = ref<'login' | 'register'>('login')
const loading = ref(false)
const errorText = ref('')
const captcha = ref<CaptchaOut | null>(null)

const form = reactive({
  username: '',
  password: '',
  password_confirm: '',
  captcha_code: '',
})

async function loadCaptcha() {
  form.captcha_code = ''
  try {
    captcha.value = await api.captcha(mode.value)
  } catch (error) {
    errorText.value = errorMessage(error)
  }
}

watch(mode, () => {
  errorText.value = ''
  loadCaptcha()
})

onMounted(loadCaptcha)

async function submit() {
  errorText.value = ''
  if (!form.username.trim() || !form.password) {
    errorText.value = '用户名和密码都不能为空'
    return
  }
  if (!form.captcha_code.trim()) {
    errorText.value = '请输入图形验证码'
    return
  }
  if (!captcha.value) {
    errorText.value = '验证码还没加载出来，点图片刷新一下'
    return
  }
  if (mode.value === 'register' && form.password !== form.password_confirm) {
    errorText.value = '两次输入的密码不一致'
    return
  }

  loading.value = true
  try {
    if (mode.value === 'register') {
      await api.register({
        username: form.username.trim(),
        password: form.password,
        password_confirm: form.password_confirm,
        captcha_id: captcha.value.captcha_id,
        captcha_code: form.captcha_code.trim(),
      })
      ElMessage.success('注册成功，请登录')
      mode.value = 'login'
      form.password = ''
      form.password_confirm = ''
      await loadCaptcha()
      return
    }

    await authStore.login({
      username: form.username.trim(),
      password: form.password,
      captcha_id: captcha.value.captcha_id,
      captcha_code: form.captcha_code.trim(),
    })
    const redirect = typeof route.query.redirect === 'string' ? route.query.redirect : '/kb'
    router.replace(redirect)
  } catch (error) {
    errorText.value = errorMessage(error)
    // 验证码是一次性的，无论因为什么失败都得换一张——
    // 不换的话用户改完密码再提交，会撞上「验证码已使用」
    await loadCaptcha()
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="login-page">
    <div class="card">
      <div class="head">
        <div class="mark">
          <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">
            <path
              d="M12 3v18M5 8c0 6 3 10 7 10s7-4 7-10"
              fill="none"
              stroke="#fff"
              stroke-width="1.8"
              stroke-linecap="round"
            />
          </svg>
        </div>
        <h1>心理知识库</h1>
        <p>基于教材的检索增强问答 · 答案带页码溯源</p>
      </div>

      <div class="tabs">
        <button
          type="button"
          class="tab"
          :class="{ 'is-on': mode === 'login' }"
          @click="mode = 'login'"
        >
          登录
        </button>
        <button
          type="button"
          class="tab"
          :class="{ 'is-on': mode === 'register' }"
          @click="mode = 'register'"
        >
          注册
        </button>
      </div>

      <form class="form" @submit.prevent="submit">
        <el-input v-model="form.username" size="large" placeholder="用户名" :prefix-icon="User" />
        <el-input
          v-model="form.password"
          size="large"
          type="password"
          placeholder="密码"
          show-password
          :prefix-icon="Lock"
        />
        <el-input
          v-if="mode === 'register'"
          v-model="form.password_confirm"
          size="large"
          type="password"
          placeholder="确认密码"
          show-password
          :prefix-icon="Lock"
        />

        <div class="captcha-row">
          <el-input
            v-model="form.captcha_code"
            size="large"
            placeholder="图形验证码"
            maxlength="6"
            @keyup.enter="submit"
          />
          <button type="button" class="captcha" title="看不清？点击换一张" @click="loadCaptcha">
            <img v-if="captcha" :src="captcha.image_base64" alt="验证码" />
            <span v-else class="captcha-empty"><el-icon><Refresh /></el-icon></span>
          </button>
        </div>

        <p v-if="errorText" class="error">{{ errorText }}</p>

        <el-button type="primary" size="large" class="submit" :loading="loading" @click="submit">
          {{ mode === 'login' ? '登录' : '注册' }}
        </el-button>
      </form>

      <p class="foot">心理学知识科普，不能替代专业诊断与治疗</p>
    </div>
  </div>
</template>

<style scoped>
.login-page {
  min-height: 100vh;
  display: grid;
  place-items: center;
  padding: 24px;
  background:
    radial-gradient(900px 420px at 50% -140px, #e3e8fb 0%, transparent 70%),
    var(--app-bg);
}

.card {
  width: 100%;
  max-width: 400px;
  background: #fff;
  border: 1px solid var(--app-border);
  border-radius: 16px;
  padding: 32px 30px 24px;
}

.head {
  text-align: center;
  margin-bottom: 22px;
}

.mark {
  width: 46px;
  height: 46px;
  border-radius: 13px;
  background: linear-gradient(140deg, #4b5fd6, #6d7ce8);
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 0 auto 12px;
}

.head h1 {
  font-size: 20px;
  font-weight: 600;
  margin: 0 0 6px;
  letter-spacing: 0.3px;
}

.head p {
  margin: 0;
  font-size: 12.5px;
  color: var(--app-text-weak);
}

.tabs {
  display: flex;
  gap: 4px;
  background: #f3f4f9;
  padding: 4px;
  border-radius: 10px;
  margin-bottom: 18px;
}

.tab {
  flex: 1;
  border: 0;
  background: transparent;
  padding: 8px 0;
  font-size: 13.5px;
  color: var(--app-text-weak);
  border-radius: 7px;
  cursor: pointer;
  transition: all 0.15s ease;
}

.tab.is-on {
  background: #fff;
  color: var(--app-text);
  font-weight: 500;
}

.form {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.captcha-row {
  display: flex;
  gap: 10px;
}

.captcha {
  flex: 0 0 116px;
  height: 40px;
  padding: 0;
  border: 1px solid var(--app-border);
  border-radius: 8px;
  background: #fafbfd;
  overflow: hidden;
  cursor: pointer;
}

.captcha img {
  width: 100%;
  height: 100%;
  /* 用 contain 而不是 cover：验证码图比例和这个框不完全一致，
     cover 会把边上的字符裁掉，而缺一个字符就等于没法输入 */
  object-fit: contain;
  display: block;
}

.captcha-empty {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: var(--app-text-weak);
}

.error {
  margin: 0;
  font-size: 12.5px;
  color: var(--el-color-danger);
  background: var(--el-color-danger-light-9);
  border: 1px solid var(--el-color-danger-light-7);
  border-radius: 8px;
  padding: 8px 10px;
}

.submit {
  width: 100%;
  margin-top: 2px;
}

.foot {
  text-align: center;
  font-size: 11.5px;
  color: var(--app-text-weak);
  margin: 18px 0 0;
}
</style>
