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
        { path: '', redirect: '/kb' },
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
        { path: 'daily', name: 'daily', component: () => import('@/views/RecommendView.vue') },
        { path: 'chat', name: 'chat', component: () => import('@/views/ChatView.vue') },
      ],
    },
    { path: '/:pathMatch(.*)*', redirect: '/kb' },
  ],
})

router.beforeEach((to) => {
  const loggedIn = Boolean(auth.token)
  if (!to.meta.public && !loggedIn) {
    return { name: 'login', query: { redirect: to.fullPath } }
  }
  if (to.name === 'login' && loggedIn) {
    return { name: 'kb-list' }
  }
  return true
})

export default router
