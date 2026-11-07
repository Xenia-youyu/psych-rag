import { createRouter, createWebHashHistory } from 'vue-router'

import { auth } from '@/api/client'
import AppLayout from '@/layouts/AppLayout.vue'

const router = createRouter({
  // hash 路由：不需要后端配 history fallback，静态托管也不会刷新 404
  history: createWebHashHistory(),
  routes: [
    {
      path: '/login',
      name: 'login',
      component: () => import('@/views/LoginView.vue'),
      meta: { public: true },
    },
    {
      path: '/',
      component: AppLayout,
      children: [
        // 问答是主入口，不再让用户先落到「知识库」再点进来
        { path: '', name: 'chat', component: () => import('@/views/ChatView.vue') },
        { path: 'chat', redirect: '/' },
        {
          path: 'favorites',
          name: 'favorites',
          component: () => import('@/views/FavoritesView.vue'),
        },
        { path: 'daily', name: 'daily', component: () => import('@/views/DailyView.vue') },
        // 语料管理（原「知识库」）：教材上传、解析进度、文本块维护
        { path: 'kb', name: 'kb-list', component: () => import('@/views/KbListView.vue') },
        {
          path: 'kb/:kbId',
          name: 'kb-detail',
          component: () => import('@/views/KbDetailView.vue'),
          props: true,
        },
        {
          path: 'kb/:kbId/docs/:docId/chunks',
          name: 'chunks',
          component: () => import('@/views/ChunksView.vue'),
          props: true,
        },
      ],
    },
    { path: '/:pathMatch(.*)*', redirect: '/' },
  ],
})

router.beforeEach((to) => {
  const loggedIn = Boolean(auth.token)
  if (!to.meta.public && !loggedIn) {
    return { name: 'login', query: { redirect: to.fullPath } }
  }
  if (to.name === 'login' && loggedIn) {
    return { name: 'chat' }
  }
  return true
})

export default router
