<script setup lang="ts">
import { onMounted } from 'vue'

import { useAuthStore } from '@/stores/auth'

const authStore = useAuthStore()

// 刷新页面后校验一次 token：过期就直接踢回登录页，
// 而不是让用户点什么都弹 401
onMounted(() => {
  authStore.refresh()
})
</script>

<template>
  <router-view />
</template>
