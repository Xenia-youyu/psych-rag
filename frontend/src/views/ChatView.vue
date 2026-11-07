<script setup lang="ts">
import {
  ArrowRight,
  ChatDotRound,
  Delete,
  Document,
  Plus,
  Promotion,
  Star,
  StarFilled,
  VideoPause,
} from '@element-plus/icons-vue'
import DOMPurify from 'dompurify'
import { ElMessage, ElMessageBox } from 'element-plus'
import { marked } from 'marked'
import { computed, nextTick, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'

import { api } from '@/api'
import { streamChat } from '@/api/chat'
import type { DonePayload } from '@/api/chat'
import { errorMessage } from '@/api/client'
import type { Citation, Session } from '@/api/types'

const route = useRoute()
const router = useRouter()

interface Bubble {
  role: 'user' | 'assistant'
  content: string
  citations: Citation[]
  streaming?: boolean
  stopped?: boolean
  error?: string
  done?: DonePayload
  hitCount?: number
  intent?: string
  strategy?: string
  favoriteId?: string
  favoriting?: boolean
  showSources?: boolean
}

const sessions = ref<Session[]>([])
const activeId = ref<string | null>(null)
const messages = ref<Bubble[]>([])
const input = ref('')
const streaming = ref(false)
const loadingHistory = ref(false)
const scrollEl = ref<HTMLElement | null>(null)
const activeCard = ref('')

let controller: AbortController | null = null
let frame = 0
let pendingText = ''

// 示例问题按「师生与家长最常问的」挑：不是最难的，是最像真人会问出口的
const suggestions = [
  '什么是皮格马利翁效应？它在教学里怎么起作用',
  '孩子考试前总说肚子疼，可能是心理原因吗',
  '如何区分正常的悲伤情绪和抑郁状态',
  '心理学里说的「信度」和「效度」分别指什么',
]

const isEmpty = computed(() => messages.value.length === 0)
const lastQuestion = computed(() => {
  for (let i = messages.value.length - 1; i >= 0; i -= 1) {
    if (messages.value[i].role === 'user') return messages.value[i].content
  }
  return ''
})

function scrollToBottom() {
  nextTick(() => {
    if (scrollEl.value) scrollEl.value.scrollTop = scrollEl.value.scrollHeight
  })
}

async function loadSessions() {
  try {
    sessions.value = await api.listSessions()
  } catch (error) {
    ElMessage.error(errorMessage(error))
  }
}

onMounted(async () => {
  await loadSessions()
  applyQuery()
  scrollToBottom()
})

// 从「每日知识」点「问一问」过来时带着 q：
// 要开的是一条新对话，不是接着上次那条聊——上下文混在一起会让模型串题
function applyQuery() {
  const question = route.query.q
  if (typeof question === 'string' && question) {
    activeId.value = null
    messages.value = []
    input.value = question
    nextTick(send)
    router.replace({ path: '/' })
  }
}

watch(
  () => route.query.q,
  () => applyQuery(),
)

async function openSession(session: Session) {
  if (streaming.value) return
  activeId.value = session.id
  loadingHistory.value = true
  try {
    const page = await api.listMessages(session.id)
    messages.value = page.items.map((item) => ({
      role: item.role,
      content: item.content,
      citations: item.citations || [],
    }))
    scrollToBottom()
  } catch (error) {
    ElMessage.error(errorMessage(error))
  } finally {
    loadingHistory.value = false
  }
}

function newChat() {
  if (streaming.value) return
  activeId.value = null
  messages.value = []
  input.value = ''
}

async function removeSession(session: Session) {
  try {
    await ElMessageBox.confirm(`删除会话「${session.title}」及其全部消息？`, '确认删除', {
      type: 'warning',
      confirmButtonText: '删除',
      cancelButtonText: '取消',
    })
  } catch {
    return
  }
  try {
    await api.deleteSession(session.id)
    if (activeId.value === session.id) newChat()
    await loadSessions()
  } catch (error) {
    ElMessage.error(errorMessage(error))
  }
}

function useSuggestion(text: string) {
  input.value = text
  send()
}

async function send() {
  const question = input.value.trim()
  if (!question || streaming.value) return

  messages.value.push({ role: 'user', content: question, citations: [] })
  const bubble: Bubble = { role: 'assistant', content: '', citations: [], streaming: true }
  messages.value.push(bubble)
  input.value = ''
  streaming.value = true
  pendingText = ''
  scrollToBottom()

  controller = new AbortController()

  try {
    await streamChat(
      {
        question,
        session_id: activeId.value,
      },
      {
        onMeta: (data) => {
          if (data.is_new) {
            activeId.value = data.session_id
            // 新会话出现在列表里，但不打断当前渲染
            loadSessions()
          }
        },
        onSources: (data) => {
          bubble.citations = data.citations || []
          bubble.hitCount = data.hit_count
          bubble.intent = data.intent
          bubble.strategy = data.strategy
        },
        onToken: (text) => {
          pendingText += text
          if (!frame) {
            // 逐 token 改响应式状态会让 markdown 每秒重算上百次
            frame = requestAnimationFrame(() => {
              frame = 0
              bubble.content = pendingText
              scrollToBottom()
            })
          }
        },
        onDone: (data) => {
          bubble.content = pendingText
          bubble.streaming = false
          bubble.done = data
          // 表格、引用卡片是在渲染后才撑开高度的，
          // 立刻滚到底会停在旧位置，留 120ms 让它排完版再滚一次
          scrollToBottom()
          window.setTimeout(scrollToBottom, 120)
        },
        onError: (data) => {
          bubble.streaming = false
          bubble.error = data.message || '生成失败'
        },
      },
      controller.signal,
    )
    if (bubble.streaming) bubble.streaming = false
  } catch (error) {
    bubble.streaming = false
    if ((error as Error).name === 'AbortError') {
      bubble.stopped = true
      bubble.content = pendingText
    } else {
      bubble.error = errorMessage(error)
    }
  } finally {
    streaming.value = false
    controller = null
    if (frame) {
      cancelAnimationFrame(frame)
      frame = 0
    }
    await loadSessions()
  }
}

function stop() {
  controller?.abort()
}

// ---------------------------------------------------------------- 收藏
async function toggleFavorite(bubble: Bubble, index: number) {
  if (bubble.favoriting) return
  if (bubble.favoriteId) {
    try {
      await api.deleteFavorite(bubble.favoriteId)
      bubble.favoriteId = ''
      ElMessage.success('已取消收藏')
    } catch (error) {
      ElMessage.error(errorMessage(error))
    }
    return
  }

  const question = questionOf(index)
  if (!question) {
    ElMessage.warning('找不到这条回答对应的问题，无法收藏')
    return
  }
  bubble.favoriting = true
  try {
    const saved = await api.createFavorite({
      question,
      answer: bubble.content,
      session_id: activeId.value || '',
      citations: bubble.citations.map((item) => ({
        sid: item.sid,
        doc_title: item.doc,
        page: item.page ?? 0,
        page_end: item.page ?? 0,
        content: item.excerpt || '',
      })),
    })
    bubble.favoriteId = saved.id
    ElMessage.success('已存入我的知识库')
  } catch (error) {
    ElMessage.error(errorMessage(error))
  } finally {
    bubble.favoriting = false
  }
}

/** 找这条回答上一条用户提问。历史消息里 role 严格交替，但不能假设顺序一定对 */
function questionOf(index: number): string {
  for (let i = index - 1; i >= 0; i -= 1) {
    if (messages.value[i].role === 'user') return messages.value[i].content
  }
  return lastQuestion.value
}

async function copyAnswer(bubble: Bubble) {
  try {
    await navigator.clipboard.writeText(bubble.content)
    ElMessage.success('已复制')
  } catch {
    ElMessage.warning('浏览器不允许访问剪贴板')
  }
}

// ---------------------------------------------------------------- 渲染
marked.setOptions({ breaks: true, gfm: true })

function renderMarkdown(text: string): string {
  const html = DOMPurify.sanitize(marked.parse(text || '') as string)
  // 先把 markdown 转完再替换引用标记，否则 [S1] 会被当成链接语法
  return html.replace(
    /\[S(\d+)\]/g,
    (_, n) => `<button type="button" class="cite" data-sid="S${n}">${n}</button>`,
  )
}

function onCiteClick(event: MouseEvent) {
  const target = (event.target as HTMLElement).closest('[data-sid]') as HTMLElement | null
  if (!target) return
  const sid = target.dataset.sid
  const holder = target.closest('.msg') as HTMLElement | null
  const index = Number(holder?.dataset.index ?? -1)
  if (!sid || index < 0) return

  const bubble = messages.value[index]
  const citation = bubble?.citations?.find((item) => item.sid === sid)
  if (!citation) {
    ElMessage.warning('这条引用的来源信息缺失')
    return
  }
  bubble.showSources = true
  activeCard.value = `${index}-${sid}`
  nextTick(() => {
    const el = document.querySelector(`[data-card="${index}-${sid}"]`)
    el?.scrollIntoView({ behavior: 'smooth', block: 'center' })
  })
}

function isCrisis(bubble: Bubble) {
  return Boolean(bubble.done?.safety)
}

function sourceLabel(citation: Citation) {
  const page = citation.page ? ` · 第 ${citation.page} 页` : ''
  return `${citation.doc || '未知来源'}${page}`
}
</script>

<template>
  <div class="chat">
    <aside class="sessions">
      <button type="button" class="new-chat" :disabled="streaming" @click="newChat">
        <el-icon><Plus /></el-icon>
        开启新对话
      </button>

      <div class="sessions-title">历史对话</div>
      <div class="session-list">
        <div v-if="!sessions.length" class="session-empty">还没有对话，从右边问一句试试</div>
        <button
          v-for="session in sessions"
          :key="session.id"
          type="button"
          class="session"
          :class="{ 'is-on': activeId === session.id }"
          @click="openSession(session)"
        >
          <div class="session-main">
            <div class="session-title">{{ session.title }}</div>
            <div class="session-meta">{{ session.message_count }} 条消息</div>
          </div>
          <el-icon class="session-del" @click.stop="removeSession(session)"><Delete /></el-icon>
        </button>
      </div>
    </aside>

    <section class="stage">
      <div ref="scrollEl" class="thread" :class="{ 'is-welcome': isEmpty }">
        <!-- 空态：这里才是产品的门面，别一上来就给个空白输入框 -->
        <div v-if="isEmpty" class="welcome">
          <div class="welcome-mark">
            <svg viewBox="0 0 24 24" width="26" height="26" aria-hidden="true">
              <path
                d="M12 3v18M5 8c0 6 3 10 7 10s7-4 7-10"
                fill="none"
                stroke="#fff"
                stroke-width="1.7"
                stroke-linecap="round"
              />
            </svg>
          </div>
          <h1>聊聊心理健康</h1>
          <p class="welcome-sub">你的专业心理学知识助手，不足之处还请多多关照哦~</p>

          <div class="welcome-input">
            <el-input
              v-model="input"
              type="textarea"
              :rows="2"
              resize="none"
              placeholder="问点什么，比如：孩子最近不想上学，可能是什么原因"
              @keydown.enter.exact.prevent="send"
            />
            <div class="welcome-actions">
              <span class="welcome-tip">Enter 发送 · Shift+Enter 换行</span>
              <el-button type="primary" :disabled="!input.trim()" @click="send">
                提问
                <el-icon class="ml"><ArrowRight /></el-icon>
              </el-button>
            </div>
          </div>

          <div class="suggestions">
            <button
              v-for="text in suggestions"
              :key="text"
              type="button"
              class="suggestion"
              @click="useSuggestion(text)"
            >
              {{ text }}
            </button>
          </div>
        </div>

        <!-- 对话态 -->
        <template v-else>
          <article
            v-for="(msg, index) in messages"
            :key="index"
            class="msg"
            :class="msg.role"
            :data-index="index"
          >
            <div v-if="msg.role === 'user'" class="bubble-user">{{ msg.content }}</div>

            <div v-else class="answer">
              <div class="answer-head">
                <span class="answer-avatar">
                  <el-icon :size="14"><ChatDotRound /></el-icon>
                </span>
                <span class="answer-label">知识库回答</span>
                <span v-if="msg.hitCount !== undefined" class="answer-meta">
                  命中 {{ msg.hitCount }} 条依据
                </span>
              </div>

              <div
                v-if="msg.content"
                class="markdown"
                v-html="renderMarkdown(msg.content)"
                @click="onCiteClick"
              />
              <div v-else-if="msg.streaming" class="thinking">正在检索教材并组织回答…</div>

              <p v-if="msg.error" class="answer-error">{{ msg.error }}</p>
              <p v-else-if="msg.stopped" class="answer-stopped">已停止生成</p>

              <!-- 引用：默认收起，避免喧宾夺主；点正文里的角标会展开并定位 -->
              <div v-if="msg.citations.length" class="sources">
                <button type="button" class="sources-toggle" @click="msg.showSources = !msg.showSources">
                  <el-icon :size="13"><Document /></el-icon>
                  {{ msg.citations.length }} 处出处
                  <span class="sources-chevron">{{ msg.showSources ? '收起' : '展开' }}</span>
                </button>
                <ol v-if="msg.showSources" class="source-list">
                  <li
                    v-for="cite in msg.citations"
                    :key="cite.sid"
                    class="source"
                    :data-card="`${index}-${cite.sid}`"
                    :class="{ 'is-active': activeCard === `${index}-${cite.sid}` }"
                  >
                    <span class="source-sid">{{ cite.sid }}</span>
                    <div class="source-body">
                      <div class="source-title">{{ sourceLabel(cite) }}</div>
                      <div class="source-excerpt">{{ cite.excerpt }}</div>
                    </div>
                  </li>
                </ol>
              </div>

              <div v-if="!msg.streaming && msg.content" class="answer-actions">
                <button
                  type="button"
                  class="act"
                  :class="{ 'is-on': msg.favoriteId }"
                  :disabled="msg.favoriting"
                  @click="toggleFavorite(msg, index)"
                >
                  <el-icon :size="14">
                    <StarFilled v-if="msg.favoriteId" />
                    <Star v-else />
                  </el-icon>
                  {{ msg.favoriteId ? '已收藏' : '收藏' }}
                </button>
                <button type="button" class="act" @click="copyAnswer(msg)">复制</button>
                <span v-if="isCrisis(msg)" class="act-crisis">已触发心理危机提示</span>
              </div>
            </div>
          </article>
        </template>
      </div>

      <footer v-if="!isEmpty" class="composer">
        <el-input
          v-model="input"
          type="textarea"
          :rows="1"
          :autosize="{ minRows: 1, maxRows: 5 }"
          resize="none"
          placeholder="继续追问，或换个问题"
          :disabled="streaming"
          @keydown.enter.exact.prevent="send"
        />
        <el-button v-if="streaming" class="send stop" @click="stop">
          <el-icon><VideoPause /></el-icon>
          停止
        </el-button>
        <el-button v-else type="primary" class="send" :disabled="!input.trim()" @click="send">
          <el-icon><Promotion /></el-icon>
        </el-button>
      </footer>
    </section>
  </div>
</template>

<style scoped>
.chat {
  display: flex;
  height: 100%;
  min-height: 0;
}

/* ------------------------------------------------------------ 会话栏 */
.sessions {
  width: 208px;
  flex: 0 0 208px;
  border-right: 1px solid var(--app-border);
  background: #fff;
  display: flex;
  flex-direction: column;
  padding: 14px 10px;
  min-height: 0;
}

.new-chat {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  width: 100%;
  padding: 9px 10px;
  border-radius: 9px;
  border: 1px dashed var(--el-color-primary-light-5);
  background: var(--el-color-primary-light-9);
  color: var(--el-color-primary-dark-2);
  font-size: 13px;
  cursor: pointer;
  transition: background 0.15s ease;
}

.new-chat:hover:not(:disabled) {
  background: var(--el-color-primary-light-8);
}

.new-chat:disabled {
  opacity: 0.55;
  cursor: not-allowed;
}

.sessions-title {
  font-size: 11.5px;
  color: var(--app-text-weak);
  padding: 16px 6px 6px;
  letter-spacing: 0.3px;
}

.session-list {
  flex: 1;
  overflow-y: auto;
  min-height: 0;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.session-empty {
  font-size: 12px;
  color: var(--app-text-weak);
  line-height: 1.7;
  padding: 8px 8px;
}

.session {
  display: flex;
  align-items: center;
  gap: 6px;
  width: 100%;
  text-align: left;
  padding: 8px 8px;
  border: 0;
  border-radius: 8px;
  background: transparent;
  cursor: pointer;
  color: var(--app-text);
}

.session:hover {
  background: #f5f6fb;
}

.session.is-on {
  background: var(--el-color-primary-light-9);
}

.session-main {
  flex: 1;
  min-width: 0;
}

.session-title {
  font-size: 12.5px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.session-meta {
  font-size: 11px;
  color: var(--app-text-weak);
  margin-top: 2px;
}

.session-del {
  opacity: 0;
  color: var(--app-text-weak);
  transition: opacity 0.15s ease;
}

.session:hover .session-del {
  opacity: 1;
}

.session-del:hover {
  color: var(--el-color-danger);
}

/* ------------------------------------------------------------ 主区 */
.stage {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.thread {
  flex: 1;
  overflow-y: auto;
  padding: 26px 32px 12px;
}

.thread.is-welcome {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0 32px 60px;
}

/* ------------------------------------------------------------ 空态 */
.welcome {
  width: 100%;
  max-width: 640px;
  text-align: center;
}

.welcome-mark {
  width: 52px;
  height: 52px;
  border-radius: 15px;
  background: linear-gradient(140deg, #4b5fd6, #6d7ce8);
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 0 auto 16px;
  box-shadow: 0 8px 20px rgba(75, 95, 214, 0.22);
}

.welcome h1 {
  font-size: 23px;
  font-weight: 600;
  margin: 0 0 8px;
  letter-spacing: 0.2px;
}

.welcome-sub {
  font-size: 13px;
  color: var(--app-text-weak);
  margin: 0 0 24px;
  line-height: 1.7;
}

.welcome-input {
  background: #fff;
  border: 1px solid var(--app-border);
  border-radius: 14px;
  padding: 12px 12px 10px;
  box-shadow: 0 4px 18px rgba(30, 40, 90, 0.05);
  text-align: left;
}

.welcome-input :deep(.el-textarea__inner) {
  border: 0;
  box-shadow: none;
  padding: 4px 6px;
  font-size: 14px;
  line-height: 1.7;
  background: transparent;
}

.welcome-actions {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 6px 4px 0;
}

.welcome-tip {
  font-size: 11.5px;
  color: var(--app-text-weak);
}

.ml {
  margin-left: 4px;
}

.suggestions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 18px;
  justify-content: center;
}

.suggestion {
  border: 1px solid var(--app-border);
  background: #fff;
  color: var(--app-text);
  font-size: 12.5px;
  padding: 7px 12px;
  border-radius: 999px;
  cursor: pointer;
  transition: all 0.15s ease;
}

.suggestion:hover {
  border-color: var(--el-color-primary-light-5);
  color: var(--el-color-primary-dark-2);
  background: var(--el-color-primary-light-9);
}

/* ------------------------------------------------------------ 消息 */
.msg {
  margin-bottom: 22px;
}

.msg.user {
  display: flex;
  justify-content: flex-end;
}

.bubble-user {
  max-width: 76%;
  background: var(--el-color-primary);
  color: #fff;
  padding: 10px 14px;
  border-radius: 14px 14px 2px 14px;
  font-size: 14px;
  line-height: 1.7;
  white-space: pre-wrap;
  word-break: break-word;
}

.answer-head {
  display: flex;
  align-items: center;
  gap: 7px;
  margin-bottom: 8px;
}

.answer-avatar {
  width: 20px;
  height: 20px;
  border-radius: 6px;
  background: var(--el-color-primary-light-9);
  color: var(--el-color-primary);
  display: flex;
  align-items: center;
  justify-content: center;
}

.answer-label {
  font-size: 12.5px;
  font-weight: 500;
  color: var(--app-text-weak);
}

.answer-meta {
  font-size: 11.5px;
  color: var(--app-text-weak);
  padding-left: 8px;
  border-left: 1px solid var(--app-border);
}

.markdown {
  font-size: 14px;
  line-height: 1.85;
  color: var(--app-text);
  word-break: break-word;
}

.markdown :deep(p) {
  margin: 0 0 10px;
}

.markdown :deep(ul),
.markdown :deep(ol) {
  margin: 0 0 10px;
  padding-left: 20px;
}

.markdown :deep(li) {
  margin-bottom: 4px;
}

.markdown :deep(h1),
.markdown :deep(h2),
.markdown :deep(h3) {
  font-size: 15px;
  margin: 14px 0 8px;
}

.markdown :deep(table) {
  border-collapse: collapse;
  margin: 8px 0 12px;
  font-size: 13px;
}

.markdown :deep(th),
.markdown :deep(td) {
  border: 1px solid var(--app-border);
  padding: 6px 10px;
}

.markdown :deep(blockquote) {
  margin: 0 0 10px;
  padding: 6px 12px;
  border-left: 3px solid var(--el-color-primary-light-7);
  background: #fafbff;
  color: var(--app-text-weak);
}

.markdown :deep(.cite) {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 17px;
  height: 17px;
  margin: 0 1px;
  padding: 0 4px;
  border: 0;
  border-radius: 5px;
  background: var(--el-color-primary-light-9);
  color: var(--el-color-primary-dark-2);
  font-size: 10.5px;
  font-weight: 600;
  vertical-align: super;
  cursor: pointer;
  line-height: 1;
}

.markdown :deep(.cite:hover) {
  background: var(--el-color-primary-light-7);
}

.thinking {
  font-size: 13px;
  color: var(--app-text-weak);
  padding: 4px 0;
}

.answer-error {
  margin: 8px 0 0;
  font-size: 13px;
  color: var(--el-color-danger);
  background: var(--el-color-danger-light-9);
  border-radius: 8px;
  padding: 8px 12px;
}

.answer-stopped {
  margin: 8px 0 0;
  font-size: 12px;
  color: var(--app-text-weak);
}

/* ------------------------------------------------------------ 引用 */
.sources {
  margin-top: 10px;
}

.sources-toggle {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  border: 1px solid var(--app-border);
  background: #fff;
  border-radius: 999px;
  padding: 4px 11px;
  font-size: 12px;
  color: var(--app-text-weak);
  cursor: pointer;
}

.sources-toggle:hover {
  border-color: var(--el-color-primary-light-5);
  color: var(--el-color-primary-dark-2);
}

.sources-chevron {
  color: var(--el-color-primary);
}

.source-list {
  list-style: none;
  margin: 10px 0 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.source {
  display: flex;
  gap: 9px;
  padding: 9px 11px;
  border-radius: 9px;
  background: #fff;
  border: 1px solid var(--app-border);
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

.source.is-active {
  border-color: var(--el-color-primary-light-5);
  box-shadow: 0 0 0 3px var(--el-color-primary-light-9);
}

.source-sid {
  flex: 0 0 auto;
  height: 18px;
  min-width: 18px;
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

.source-body {
  min-width: 0;
}

.source-title {
  font-size: 12.5px;
  font-weight: 500;
  color: var(--app-text);
}

.source-excerpt {
  font-size: 12px;
  color: var(--app-text-weak);
  line-height: 1.65;
  margin-top: 3px;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

/* ------------------------------------------------------------ 操作 */
.answer-actions {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-top: 10px;
}

.act {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  border: 1px solid transparent;
  background: transparent;
  color: var(--app-text-weak);
  font-size: 12px;
  padding: 4px 9px;
  border-radius: 7px;
  cursor: pointer;
}

.act:hover {
  background: #f5f6fb;
  color: var(--app-text);
}

.act.is-on {
  color: #d98b12;
  background: #fff8e8;
}

.act-crisis {
  font-size: 11.5px;
  color: var(--el-color-danger);
  margin-left: 4px;
}

/* ------------------------------------------------------------ 输入区 */
.composer {
  display: flex;
  align-items: flex-end;
  gap: 10px;
  padding: 12px 32px 20px;
  border-top: 1px solid var(--app-border);
  background: #fff;
}

.composer :deep(.el-textarea__inner) {
  border-radius: 12px;
  padding: 10px 13px;
  font-size: 14px;
  line-height: 1.7;
  box-shadow: 0 0 0 1px var(--app-border) inset;
  background: #fbfbfe;
}

.composer :deep(.el-textarea__inner:focus) {
  box-shadow: 0 0 0 1px var(--el-color-primary) inset;
  background: #fff;
}

.send {
  height: 40px;
  padding: 0 16px;
  border-radius: 11px;
  flex: 0 0 auto;
}

.send.stop {
  color: var(--el-color-danger);
  border-color: var(--el-color-danger-light-5);
}
</style>
