<script setup lang="ts">
import { Collection, Delete, EditPen, Search } from '@element-plus/icons-vue'
import DOMPurify from 'dompurify'
import { ElMessage, ElMessageBox } from 'element-plus'
import { marked } from 'marked'
import { computed, onMounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'

import { api } from '@/api'
import { errorMessage } from '@/api/client'
import type { Favorite } from '@/api/types'

const router = useRouter()

const items = ref<Favorite[]>([])
const total = ref(0)
const page = ref(1)
const size = 10
const keyword = ref('')
const loading = ref(false)
const expanded = ref<string | null>(null)
const editing = ref<string | null>(null)
const draftNote = ref('')

const isEmpty = computed(() => !loading.value && items.value.length === 0)

marked.setOptions({ breaks: true, gfm: true })

function render(text: string): string {
  return DOMPurify.sanitize(marked.parse(text || '') as string)
}

async function load() {
  loading.value = true
  try {
    const data = await api.listFavorites({ q: keyword.value.trim(), page: page.value, size })
    items.value = data.items
    total.value = data.total
  } catch (error) {
    ElMessage.error(errorMessage(error))
  } finally {
    loading.value = false
  }
}

onMounted(load)

let timer = 0
watch(keyword, () => {
  // 输入时不必每个字符都打一次接口
  window.clearTimeout(timer)
  timer = window.setTimeout(() => {
    page.value = 1
    load()
  }, 320)
})

function toggle(item: Favorite) {
  expanded.value = expanded.value === item.id ? null : item.id
  editing.value = null
}

function startEdit(item: Favorite) {
  editing.value = item.id
  draftNote.value = item.note
}

async function saveNote(item: Favorite) {
  try {
    const saved = await api.updateFavorite(item.id, draftNote.value.trim())
    item.note = saved.note
    editing.value = null
    ElMessage.success('备注已保存')
  } catch (error) {
    ElMessage.error(errorMessage(error))
  }
}

async function remove(item: Favorite) {
  try {
    await ElMessageBox.confirm('从我的知识库里移除这条内容？', '确认移除', {
      type: 'warning',
      confirmButtonText: '移除',
      cancelButtonText: '取消',
    })
  } catch {
    return
  }
  try {
    await api.deleteFavorite(item.id)
    ElMessage.success('已移除')
    if (items.value.length === 1 && page.value > 1) page.value -= 1
    await load()
  } catch (error) {
    ElMessage.error(errorMessage(error))
  }
}

function askAgain(item: Favorite) {
  router.push({ path: '/', query: { q: item.question } })
}

function formatDate(value?: string) {
  if (!value) return ''
  const d = new Date(value)
  if (Number.isNaN(d.getTime())) return ''
  return `${d.getMonth() + 1} 月 ${d.getDate()} 日`
}

function onPage(value: number) {
  page.value = value
  load()
}
</script>

<template>
  <div class="page">
    <header class="head">
      <div>
        <h1>我的知识库</h1>
        <p class="sub">从问答里收藏下来的知识与观点，按收藏时间倒序</p>
      </div>
      <div class="head-search">
        <el-input v-model="keyword" placeholder="搜问题、答案或书名" clearable :prefix-icon="Search" />
      </div>
    </header>

    <div v-if="isEmpty" class="empty">
      <el-icon :size="34"><Collection /></el-icon>
      <p class="empty-title">{{ keyword ? '没有匹配的收藏' : '还没有收藏任何内容' }}</p>
      <p class="empty-sub">
        {{
          keyword
            ? '换个关键词试试，搜索会匹配问题、答案正文和来源书名'
            : '在智能问答里看到有用的回答，点下方的「收藏」，它就会出现在这里'
        }}
      </p>
      <el-button v-if="!keyword" type="primary" @click="router.push('/')">去提问</el-button>
    </div>

    <template v-else>
      <div v-loading="loading" class="list">
        <article
          v-for="item in items"
          :key="item.id"
          class="card"
          :class="{ 'is-open': expanded === item.id }"
        >
          <div class="card-head" @click="toggle(item)">
            <div class="card-main">
              <div class="card-question">{{ item.question }}</div>
              <div class="card-meta">
                <span v-if="item.source" class="tag-source">{{ item.source }}</span>
                <span class="card-date">{{ formatDate(item.created_at) }}</span>
                <span v-if="item.citations.length" class="card-cites">
                  {{ item.citations.length }} 处出处
                </span>
              </div>
            </div>
            <span class="card-toggle">{{ expanded === item.id ? '收起' : '展开' }}</span>
          </div>

          <p v-if="item.note && expanded !== item.id" class="card-note">
            备注：{{ item.note }}
          </p>

          <div v-if="expanded === item.id" class="card-body">
            <div class="markdown" v-html="render(item.answer)" />

            <div v-if="item.citations.length" class="cites">
              <div class="cites-title">出处</div>
              <div v-for="cite in item.citations" :key="cite.sid" class="cite-row">
                <span class="cite-sid">{{ cite.sid }}</span>
                <div>
                  <div class="cite-title">
                    {{ cite.doc_title }}
                    <template v-if="cite.page"> · 第 {{ cite.page }} 页</template>
                  </div>
                  <div class="cite-text">{{ cite.content }}</div>
                </div>
              </div>
            </div>

            <div class="note-area">
              <template v-if="editing === item.id">
                <el-input
                  v-model="draftNote"
                  type="textarea"
                  :rows="2"
                  maxlength="500"
                  show-word-limit
                  placeholder="写点自己的理解、疑问或应用场景"
                />
                <div class="note-actions">
                  <el-button size="small" @click="editing = null">取消</el-button>
                  <el-button size="small" type="primary" @click="saveNote(item)">保存备注</el-button>
                </div>
              </template>
              <template v-else>
                <p v-if="item.note" class="note-text">{{ item.note }}</p>
                <el-button size="small" :icon="EditPen" @click="startEdit(item)">
                  {{ item.note ? '改备注' : '加备注' }}
                </el-button>
              </template>
            </div>

            <div class="card-foot">
              <el-button size="small" @click="askAgain(item)">就这个问题再问一次</el-button>
              <el-button size="small" text :icon="Delete" @click="remove(item)">移除</el-button>
            </div>
          </div>
        </article>
      </div>

      <div v-if="total > size" class="pager">
        <el-pagination
          layout="prev, pager, next"
          :current-page="page"
          :page-size="size"
          :total="total"
          background
          @current-change="onPage"
        />
      </div>
    </template>
  </div>
</template>

<style scoped>
.page {
  max-width: 880px;
  margin: 0 auto;
  padding: 30px 32px 60px;
}

.head {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 20px;
  margin-bottom: 22px;
}

.head h1 {
  font-size: 20px;
  font-weight: 600;
  margin: 0 0 6px;
}

.sub {
  font-size: 12.5px;
  color: var(--app-text-weak);
  margin: 0;
}

.head-search {
  width: 260px;
  flex: 0 0 auto;
}

.empty {
  text-align: center;
  padding: 70px 20px;
  color: var(--app-text-weak);
}

.empty-title {
  font-size: 15px;
  color: var(--app-text);
  margin: 14px 0 6px;
}

.empty-sub {
  font-size: 12.5px;
  line-height: 1.8;
  margin: 0 0 18px;
}

.list {
  display: flex;
  flex-direction: column;
  gap: 10px;
  min-height: 120px;
}

.card {
  background: #fff;
  border: 1px solid var(--app-border);
  border-radius: 12px;
  padding: 14px 16px;
  transition: border-color 0.18s ease, box-shadow 0.18s ease;
}

.card.is-open {
  border-color: var(--el-color-primary-light-5);
  box-shadow: 0 4px 18px rgba(30, 40, 90, 0.05);
}

.card-head {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  cursor: pointer;
}

.card-main {
  flex: 1;
  min-width: 0;
}

.card-question {
  font-size: 14.5px;
  font-weight: 500;
  line-height: 1.6;
  color: var(--app-text);
}

.card-meta {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-top: 6px;
  font-size: 11.5px;
  color: var(--app-text-weak);
}

.tag-source {
  background: var(--el-color-primary-light-9);
  color: var(--el-color-primary-dark-2);
  border-radius: 5px;
  padding: 1px 7px;
  max-width: 260px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.card-toggle {
  font-size: 12px;
  color: var(--el-color-primary);
  flex: 0 0 auto;
  padding-top: 2px;
}

.card-note {
  margin: 10px 0 0;
  font-size: 12.5px;
  color: var(--app-text-weak);
  background: #fafbff;
  border-radius: 8px;
  padding: 8px 10px;
  line-height: 1.7;
}

.card-body {
  margin-top: 14px;
  padding-top: 14px;
  border-top: 1px dashed var(--app-border);
}

.markdown {
  font-size: 13.5px;
  line-height: 1.85;
  color: var(--app-text);
  word-break: break-word;
}

.markdown :deep(p) {
  margin: 0 0 9px;
}

.markdown :deep(ul),
.markdown :deep(ol) {
  margin: 0 0 9px;
  padding-left: 20px;
}

.markdown :deep(table) {
  border-collapse: collapse;
  margin: 8px 0;
  font-size: 12.5px;
}

.markdown :deep(th),
.markdown :deep(td) {
  border: 1px solid var(--app-border);
  padding: 5px 9px;
}

.cites {
  margin-top: 14px;
  display: flex;
  flex-direction: column;
  gap: 7px;
}

.cites-title {
  font-size: 11.5px;
  color: var(--app-text-weak);
}

.cite-row {
  display: flex;
  gap: 8px;
  background: #fafbff;
  border-radius: 8px;
  padding: 8px 10px;
}

.cite-sid {
  flex: 0 0 auto;
  height: 17px;
  min-width: 17px;
  padding: 0 4px;
  border-radius: 5px;
  background: var(--el-color-primary-light-9);
  color: var(--el-color-primary-dark-2);
  font-size: 10.5px;
  font-weight: 600;
  display: flex;
  align-items: center;
  justify-content: center;
}

.cite-title {
  font-size: 12px;
  font-weight: 500;
}

.cite-text {
  font-size: 11.5px;
  color: var(--app-text-weak);
  line-height: 1.65;
  margin-top: 2px;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.note-area {
  margin-top: 14px;
}

.note-text {
  margin: 0 0 8px;
  font-size: 12.5px;
  line-height: 1.75;
  color: var(--app-text);
  background: #fffaf0;
  border-left: 3px solid #f0c674;
  padding: 8px 10px;
  border-radius: 0 8px 8px 0;
}

.note-actions {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
  margin-top: 8px;
}

.card-foot {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 14px;
}

.pager {
  display: flex;
  justify-content: center;
  margin-top: 22px;
}
</style>
