<script setup lang="ts">
import { ChatDotRound, Collection, Reading, SwitchButton } from '@element-plus/icons-vue'
import { computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'

import { useAuthStore } from '@/stores/auth'

const route = useRoute()
const router = useRouter()
const authStore = useAuthStore()

const active = computed(() => {
  if (route.path.startsWith('/kb')) return 'kb'
  if (route.path.startsWith('/daily')) return 'daily'
  if (route.path.startsWith('/chat')) return 'chat'
  return ''
})

const items = [
  { key: 'kb', label: '知识库', hint: '文档与文本块', icon: Collection, to: '/kb' },
  { key: 'daily', label: '每日知识', hint: '随机推荐 · 可追问', icon: Reading, to: '/daily' },
  { key: 'chat', label: '智能问答', hint: '检索增强回答', icon: ChatDotRound, to: '/chat' },
]

function logout() {
  authStore.logout()
  router.push({ name: 'login' })
}
</script>

<template>
  <div class="shell">
    <aside class="side">
      <div class="brand">
        <div class="mark">
          <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true">
            <path
              d="M12 3v18M5 8c0 6 3 10 7 10s7-4 7-10"
              fill="none"
              stroke="#fff"
              stroke-width="1.8"
              stroke-linecap="round"
            />
          </svg>
        </div>
        <div>
          <div class="brand-name">心理知识库</div>
          <div class="brand-sub">RAG 问答系统</div>
        </div>
      </div>

      <nav class="nav">
        <router-link
          v-for="item in items"
          :key="item.key"
          :to="item.to"
          class="nav-item"
          :class="{ 'is-active': active === item.key }"
        >
          <el-icon :size="17"><component :is="item.icon" /></el-icon>
          <span class="nav-text">
            <span class="nav-label">{{ item.label }}</span>
            <span class="nav-hint">{{ item.hint }}</span>
          </span>
        </router-link>
      </nav>

      <div class="side-foot">
        <div class="who">
          <div class="avatar">{{ (authStore.user?.username || '?').slice(0, 1).toUpperCase() }}</div>
          <div class="who-text">
            <div class="who-name">{{ authStore.user?.username || '未登录' }}</div>
            <div class="who-role">普通用户</div>
          </div>
        </div>
        <el-button text size="small" :icon="SwitchButton" @click="logout">退出</el-button>
      </div>
    </aside>

    <main class="main">
      <router-view />
    </main>
  </div>
</template>

<style scoped>
.shell {
  display: flex;
  height: 100%;
  background: var(--app-bg);
}

.side {
  width: 236px;
  flex: 0 0 236px;
  background: #fff;
  border-right: 1px solid var(--app-border);
  display: flex;
  flex-direction: column;
  padding: 18px 14px 14px;
}

.brand {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 4px 8px 18px;
}

.mark {
  width: 34px;
  height: 34px;
  border-radius: 10px;
  background: linear-gradient(140deg, #4b5fd6, #6d7ce8);
  display: flex;
  align-items: center;
  justify-content: center;
}

.brand-name {
  font-size: 14px;
  font-weight: 600;
  letter-spacing: 0.2px;
}

.brand-sub {
  font-size: 11px;
  color: var(--app-text-weak);
  margin-top: 2px;
}

.nav {
  display: flex;
  flex-direction: column;
  gap: 4px;
  flex: 1;
  min-height: 0;
}

.nav-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 9px 10px;
  border-radius: 9px;
  color: var(--app-text);
  transition: background 0.15s ease;
}

.nav-item:hover {
  background: #f4f5fa;
}

.nav-item.is-active {
  background: var(--el-color-primary-light-9);
  color: var(--el-color-primary-dark-2);
}

.nav-text {
  display: flex;
  flex-direction: column;
  line-height: 1.25;
}

.nav-label {
  font-size: 13.5px;
  font-weight: 500;
}

.nav-hint {
  font-size: 11px;
  color: var(--app-text-weak);
}

.nav-item.is-active .nav-hint {
  color: var(--el-color-primary);
}

.side-foot {
  border-top: 1px solid var(--app-border);
  padding: 12px 0 2px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  flex: 0 0 auto;
}

.who {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
}

.avatar {
  width: 28px;
  height: 28px;
  border-radius: 50%;
  background: var(--el-color-primary-light-8);
  color: var(--el-color-primary-dark-2);
  font-size: 12px;
  font-weight: 600;
  display: flex;
  align-items: center;
  justify-content: center;
  flex: 0 0 28px;
}

.who-text {
  min-width: 0;
}

.who-name {
  font-size: 12.5px;
  font-weight: 500;
  line-height: 1.35;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.who-role {
  font-size: 11px;
  line-height: 1.35;
  color: var(--app-text-weak);
}

.main {
  flex: 1;
  min-width: 0;
  height: 100%;
  overflow: auto;
}
</style>
