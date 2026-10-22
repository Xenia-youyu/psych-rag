<script setup lang="ts">
import { ChatDotRound, Delete, Plus, Promotion, VideoPause } from '@element-plus/icons-vue'
import DOMPurify from 'dompurify'
import { ElMessage, ElMessageBox } from 'element-plus'
import { marked } from 'marked'
import { computed, nextTick, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'

import { api } from '@/api'
import { streamChat } from '@/api/chat'
import type { DonePayload } from '@/api/chat'
import { errorMessage } from '@/api/client'
import type { Citation, Kb, Session } from '@/api/types'

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
}

const sessions = ref<Session[]>([])
const kbs = ref<Kb[]>([])
const activeId = ref<string | null>(null)
const messages = ref<Bubble[]>([])
const selectedKbIds = ref<string[]>([])
const input = ref('')
const streaming = ref(false)
const loadingHistory = ref(false)
const scrollEl = ref<HTMLElement | null>(null)

let controller: AbortController | null = null
let frame = 0
let pendingText = ''

const suggestions = [
  '实验中的自变量和因变量怎么区分',
  '儿童认知发展阶段有哪些',
  '心理学论文写作的格式规范',
]

const canSend = computed(() => !streaming.value && input.value.trim().length > 0)

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

async function loadKbs() {
  try {
    kbs.value = await api.listKbs()
    if (!selectedKbIds.value.length && kbs.value.length) {
      selectedKbIds.value = [kbs.value[0].id]
    }
  } catch (error) {
    ElMessage.error(errorMessage(error))
  }
}

// 从「每日知识」点「就此提问」过来时带着 q 和 kb：
// 要开的是一条新对话，不是接着上次那条聊——上下文混在一起会让模型串题
function applyQuery() {
  const kb = route.query.kb
  const question = route.query.q
  if (typeof kb === 'string' && kb) {
    selectedKbIds.value = [kb]
  }
  if (typeof question === 'string' && question) {
    activeId.value = null
    messages.value = []
    input.value = question
  }
}

onMounted(async () => {
  await Promise.all([loadSessions(), loadKbs()])
  applyQuery()
  scrollToBottom()
})

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
    if (session.kb_ids?.length) selectedKbIds.value = [...session.kb_ids]
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
  if (!selectedKbIds.value.length) {
    ElMessage.warning('请先在右上角选择至少一个知识库')
    return
  }

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
        kb_ids: selectedKbIds.value,
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

// ---------------------------------------------------------------- 渲染
marked.setOptions({ breaks: true, gfm: true })

function renderMarkdown(text: string): string {
  const html = DOMPurify.sanitize(marked.parse(text || '') as string)
  // 先把 markdown 转完再替换引用标记，否则 [S1] 会被当成链接语法
  return html.replace(
    /\[S(\d+)\]/g,
    (_, n) => `<button type="button" class="cite" data-sid="S${n}">S${n}</button>`,
  )
}

function onCiteClick(event: MouseEvent) {
  const target = (event.target as HTMLElement).closest('[data-sid]') as HTMLElement | null
  if (!target) return
  const sid = target.dataset.sid
  const holder = target.closest('.msg') as HTMLElement | null
  const index = Number(holder?.dataset.index ?? -1)
  if (!sid || index < 0) return

  const citation = messages.value[index]?.citations?.find((item) => item.sid === sid)
  if (!citation) {
    ElMessage.warning('这条引用的来源信息缺失')
    return
  }
  if (!citation.doc_id) {
    ElMessage.warning('该引用缺少文档定位信息，无法跳转')
    return
  }
  const kbId = selectedKbIds.value[0]
  if (!kbId) {
    ElMessage.warning('请先选择知识库')
    return
  }
  router.push({
    name: 'chunks',
    params: { kbId, docId: citation.doc_id },
    query: citation.page ? { page_no: String(citation.page) } : {},
  })
}

function isCrisis(bubble: Bubble) {
  return Boolean(bubble.done?.safety)
}
</script>

<template>
  <div class="chat">
    <aside class="sessions">
      <div class="sessions-head">
        <span class="sessions-title">会话</span>
        <el-button text size="small" :icon="Plus" @click="newChat">新对话</el-button>
      </div>

      <div class="session-list">
        <el-empty v-if="!sessions.length" description="还没有会话" :image-size="54" />

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

    <section class="thread">
      <header class="thread-head">
        <div class="thread-title">
          <el-icon class="thread-icon"><ChatDotRound /></el-icon>
          <span>{{ activeId ? '继续对话' : '新对话' }}</span>
        </div>
        <div class="kb-picker">
          <span class="picker-label">检索范围</span>
          <el-select
            v-model="selectedKbIds"
            multiple
            collapse-tags
            collapse-tags-tooltip
            placeholder="选择知识库"
            size="small"
            style="min-width: 210px"
            :disabled="streaming"
          >
            <el-option v-for="kb in kbs" :key="kb.id" :label="kb.name" :value="kb.id" />
          </el-select>
        </div>
      </header>

      <div ref="scrollEl" v-loading="loadingHistory" class="messages">
        <div v-if="!messages.length" class="welcome">
          <div class="welcome-icon">
            <el-icon :size="26"><ChatDotRound /></el-icon>
          </div>
          <h3>基于教材提问，答案带页码</h3>
          <p class="muted">
            回答只依据所选知识库里的内容。检索不到时会直接说「未覆盖」，不会硬编一个。
          </p>
          <div class="suggestions">
            <button
              v-for="item in suggestions"
              :key="item"
              type="button"
              class="suggestion"
              @click="useSuggestion(item)"
            >
              {{ item }}
            </button>
          </div>
        </div>

        <div
          v-for="(bubble, index) in messages"
          :key="index"
          class="msg"
          :class="[`is-${bubble.role}`, { 'is-crisis': isCrisis(bubble) }]"
          :data-index="index"
        >
          <div class="avatar" :class="bubble.role">
            {{ bubble.role === 'user' ? '我' : 'AI' }}
          </div>

          <div class="bubble-wrap">
            <div v-if="bubble.role === 'user'" class="bubble user-bubble">{{ bubble.content }}</div>

            <template v-else>
              <div
                v-if="bubble.content"
                class="bubble ai-bubble markdown"
                @click="onCiteClick"
                v-html="renderMarkdown(bubble.content)"
              />

              <div v-else-if="bubble.streaming" class="thinking">
                <span class="dot" /><span class="dot" /><span class="dot" />
                正在检索并组织答案
              </div>

              <div v-if="bubble.error" class="bubble error-bubble">
                <strong>生成失败</strong>
                <p>{{ bubble.error }}</p>
              </div>

              <div v-if="bubble.stopped" class="stopped">已停止生成，后面的内容没有输出</div>

              <div v-if="bubble.citations.length" class="cites">
                <div class="cites-head">
                  引用 {{ bubble.citations.length }} 处
                  <span v-if="bubble.hitCount !== undefined" class="muted">
                    · 命中 {{ bubble.hitCount }} 条
                  </span>
                  <span v-if="bubble.intent" class="muted"> · 意图 {{ bubble.intent }}</span>
                </div>
                <div class="cite-list">
                  <button
                    v-for="citation in bubble.citations"
                    :key="citation.sid"
                    type="button"
                    class="cite-card"
                    @click="onCiteClick"
                    :data-sid="citation.sid"
                  >
                    <span class="cite-sid">{{ citation.sid }}</span>
                    <span class="cite-body">
                      <span class="cite-doc">{{ citation.doc || '未知来源' }}</span>
                      <span class="cite-page mono" v-if="citation.page">第 {{ citation.page }} 页</span>
                      <span class="cite-excerpt">{{ citation.excerpt }}</span>
                    </span>
                  </button>
                </div>
              </div>

              <div v-if="bubble.done?.no_context && !bubble.done?.safety" class="hint-line">
                知识库没有覆盖到这部分内容，回答只基于已有材料。
              </div>

              <div v-if="bubble.done?.safety" class="safety-line">
                这是安全提示的固定文案，不构成诊断或建议。如有困扰，请联系学校心理中心或专业机构。
              </div>

              <div v-if="bubble.done?.dangling?.length" class="dangling-line">
                以下引用编号在当前来源里找不到：
                <span class="mono">{{ bubble.done.dangling.join('、') }}</span>
              </div>

              <div v-if="bubble.done" class="msg-meta mono">
                {{ bubble.done.text_len }} 字 · {{ (bubble.done.latency_ms / 1000).toFixed(1) }}s ·
                tokens {{ bubble.done.tokens_in }}+{{ bubble.done.tokens_out }}
              </div>
            </template>
          </div>
        </div>
      </div>

      <footer class="composer">
        <div class="composer-box">
          <el-input
            v-model="input"
            type="textarea"
            :rows="2"
            resize="none"
            maxlength="1000"
            show-word-limit
            placeholder="问一个教材里能查到的问题，Enter 发送，Shift+Enter 换行"
            @keydown.enter.exact.prevent="send"
          />
          <div class="composer-actions">
            <el-button v-if="streaming" :icon="VideoPause" @click="stop">停止</el-button>
            <el-button type="primary" :icon="Promotion" :disabled="!canSend" @click="send">
              发送
            </el-button>
          </div>
        </div>
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

.sessions {
  width: 228px;
  flex: 0 0 228px;
  border-right: 1px solid var(--app-border);
  background: #fff;
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.sessions-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 12px 10px 16px;
}

.sessions-title {
  font-size: 12px;
  color: var(--app-text-weak);
  letter-spacing: 0.6px;
}

.session-list {
  flex: 1;
  overflow: auto;
  padding: 0 8px 12px;
  display: flex;
  flex-direction: column;
  gap: 3px;
}

.session {
  display: flex;
  align-items: center;
  gap: 6px;
  width: 100%;
  border: 0;
  background: transparent;
  text-align: left;
  padding: 9px 10px;
  border-radius: 8px;
  cursor: pointer;
}

.session:hover {
  background: #f5f6fa;
}

.session.is-on {
  background: var(--el-color-primary-light-9);
}

.session-main {
  min-width: 0;
  flex: 1;
}

.session-title {
  font-size: 13px;
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
}

.session:hover .session-del {
  opacity: 1;
}

.thread {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.thread-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 12px 20px;
  border-bottom: 1px solid var(--app-border);
  background: #fff;
}

.thread-title {
  display: flex;
  align-items: center;
  gap: 7px;
  font-size: 13.5px;
  font-weight: 500;
}

.thread-icon {
  color: var(--el-color-primary);
}

.kb-picker {
  display: flex;
  align-items: center;
  gap: 8px;
  flex: 0 0 auto;
}

.picker-label {
  font-size: 12px;
  color: var(--app-text-weak);
  white-space: nowrap;
}

.messages {
  flex: 1;
  overflow: auto;
  padding: 20px 24px 8px;
  min-height: 0;
}

.welcome {
  max-width: 520px;
  margin: 60px auto 0;
  text-align: center;
}

.welcome-icon {
  width: 52px;
  height: 52px;
  border-radius: 14px;
  background: var(--el-color-primary-light-9);
  color: var(--el-color-primary);
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 0 auto 14px;
}

.welcome h3 {
  font-size: 16px;
  font-weight: 600;
  margin: 0 0 8px;
}

.welcome p {
  font-size: 12.5px;
  line-height: 1.6;
  margin: 0 0 18px;
}

.suggestions {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.suggestion {
  border: 1px solid var(--app-border);
  background: #fff;
  border-radius: 9px;
  padding: 10px 14px;
  font-size: 13px;
  color: var(--app-text);
  cursor: pointer;
  text-align: left;
}

.suggestion:hover {
  border-color: var(--el-color-primary-light-5);
  color: var(--el-color-primary-dark-2);
}

.msg {
  display: flex;
  gap: 11px;
  margin-bottom: 20px;
}

.msg.is-user {
  flex-direction: row-reverse;
}

.avatar {
  width: 28px;
  height: 28px;
  flex: 0 0 28px;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 11.5px;
  font-weight: 600;
}

.avatar.user {
  background: #eef0f6;
  color: #5b6472;
}

.avatar.assistant {
  background: linear-gradient(140deg, #4b5fd6, #6d7ce8);
  color: #fff;
}

.bubble-wrap {
  /* 回答里常有四列对比表格，720px 会把列名挤成两三个字一行 */
  max-width: min(860px, 82%);
  min-width: 0;
}

.msg.is-user .bubble-wrap {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
}

.bubble {
  border-radius: 12px;
  padding: 10px 14px;
  font-size: 13.5px;
  line-height: 1.72;
  word-break: break-word;
}

.user-bubble {
  background: var(--el-color-primary);
  color: #fff;
  white-space: pre-wrap;
}

.ai-bubble {
  background: #fff;
  border: 1px solid var(--app-border);
}

.msg.is-crisis .ai-bubble {
  border-color: var(--el-color-warning-light-5);
  background: var(--el-color-warning-light-9);
}

.error-bubble {
  background: var(--el-color-danger-light-9);
  border: 1px solid var(--el-color-danger-light-7);
  color: var(--el-color-danger-dark-2);
}

.error-bubble p {
  margin: 4px 0 0;
  font-size: 12.5px;
}

.thinking {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: 12.5px;
  color: var(--app-text-weak);
  background: #fff;
  border: 1px solid var(--app-border);
  border-radius: 12px;
  padding: 10px 14px;
}

.dot {
  width: 5px;
  height: 5px;
  border-radius: 50%;
  background: var(--el-color-primary);
  animation: blink 1.1s infinite ease-in-out;
}

.dot:nth-child(2) {
  animation-delay: 0.18s;
}

.dot:nth-child(3) {
  animation-delay: 0.36s;
}

@keyframes blink {
  0%,
  80%,
  100% {
    opacity: 0.25;
  }
  40% {
    opacity: 1;
  }
}

.stopped {
  font-size: 11.5px;
  color: var(--app-text-weak);
  margin-top: 6px;
}

.cites {
  margin-top: 10px;
}

.cites-head {
  font-size: 11.5px;
  color: var(--app-text-weak);
  margin-bottom: 6px;
}

.cite-list {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.cite-card {
  display: flex;
  gap: 9px;
  align-items: flex-start;
  width: 100%;
  text-align: left;
  border: 1px solid var(--app-border);
  background: #fafbfd;
  border-radius: 9px;
  padding: 8px 10px;
  cursor: pointer;
}

.cite-card:hover {
  border-color: var(--el-color-primary-light-5);
  background: #fff;
}

.cite-sid {
  flex: 0 0 auto;
  font-size: 10.5px;
  font-weight: 600;
  color: var(--el-color-primary);
  background: var(--el-color-primary-light-9);
  border-radius: 4px;
  padding: 1px 5px;
  margin-top: 1px;
}

.cite-body {
  min-width: 0;
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 6px;
}

.cite-doc {
  font-size: 12.5px;
  font-weight: 500;
}

.cite-page {
  color: var(--app-text-weak);
}

.cite-excerpt {
  flex: 1 0 100%;
  font-size: 11.5px;
  color: var(--app-text-weak);
  overflow: hidden;
  text-overflow: ellipsis;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
}

.hint-line,
.dangling-line {
  font-size: 11.5px;
  margin-top: 8px;
  padding: 7px 10px;
  border-radius: 7px;
}

.hint-line {
  color: #7a5b12;
  background: var(--el-color-warning-light-9);
}

.dangling-line {
  color: var(--el-color-danger-dark-2);
  background: var(--el-color-danger-light-9);
}

.safety-line {
  font-size: 11.5px;
  color: #7a5b12;
  background: var(--el-color-warning-light-8);
  border-radius: 7px;
  padding: 7px 10px;
  margin-top: 8px;
}

.msg-meta {
  color: var(--app-text-weak);
  margin-top: 8px;
}

.composer {
  padding: 12px 20px 18px;
  background: #fff;
  border-top: 1px solid var(--app-border);
}

.composer-box {
  max-width: 860px;
  margin: 0 auto;
}

.composer-actions {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
  margin-top: 8px;
}

/* markdown 内容排版：默认的 h1/h2 在气泡里太大 */
.markdown :deep(h1),
.markdown :deep(h2),
.markdown :deep(h3) {
  font-size: 14px;
  font-weight: 600;
  margin: 12px 0 6px;
}

.markdown :deep(h1:first-child),
.markdown :deep(h2:first-child),
.markdown :deep(h3:first-child) {
  margin-top: 0;
}

.markdown :deep(p) {
  margin: 0 0 8px;
}

.markdown :deep(p:last-child) {
  margin-bottom: 0;
}

.markdown :deep(ul),
.markdown :deep(ol) {
  margin: 4px 0 8px;
  padding-left: 20px;
}

.markdown :deep(li) {
  margin: 2px 0;
}

.markdown :deep(table) {
  border-collapse: collapse;
  width: 100%;
  margin: 8px 0;
  font-size: 12.5px;
}

.markdown :deep(th),
.markdown :deep(td) {
  border: 1px solid var(--app-border);
  padding: 6px 8px;
  text-align: left;
  vertical-align: top;
}

.markdown :deep(th) {
  background: #f7f8fb;
  font-weight: 600;
}

.markdown :deep(code) {
  background: #f4f5f9;
  border-radius: 4px;
  padding: 1px 4px;
  font-size: 12px;
}

.markdown :deep(blockquote) {
  margin: 8px 0;
  padding-left: 10px;
  border-left: 3px solid var(--el-color-primary-light-7);
  color: var(--app-text-weak);
}

.markdown :deep(.cite) {
  display: inline-block;
  border: 0;
  background: var(--el-color-primary-light-9);
  color: var(--el-color-primary-dark-2);
  font-size: 10.5px;
  font-weight: 600;
  border-radius: 4px;
  padding: 0 4px;
  margin: 0 1px;
  cursor: pointer;
  vertical-align: baseline;
}

.markdown :deep(.cite:hover) {
  background: var(--el-color-primary-light-7);
}
</style>
