<script setup lang="ts">
import { ArrowLeft, Delete, Edit, Refresh, Search } from '@element-plus/icons-vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { nextTick, onMounted, reactive, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'

import { api } from '@/api'
import { errorMessage } from '@/api/client'
import type { Chunk, Doc } from '@/api/types'

const props = defineProps<{ kbId: string; docId: string }>()
const route = useRoute()
const router = useRouter()

// 从问答页点引用跳过来时带着页码，高亮那一页的块
const focusPage = ref<number | null>(
  route.query.page_no && !Number.isNaN(Number(route.query.page_no))
    ? Number(route.query.page_no)
    : null,
)

const doc = ref<Doc | null>(null)
const chunks = ref<Chunk[]>([])
const total = ref(0)
const loading = ref(false)
const pending = ref(0)

const query = reactive({ page: 1, size: 20, keyword: '', only_enabled: false })

const dialogVisible = ref(false)
const saving = ref(false)
const editing = ref<Chunk | null>(null)
const draft = ref('')

// 一条 300 字的块展开就 400px 高，20 条一页能把屏幕拉成上万像素。
// 默认折到 5 行，点开看全文。
const expanded = ref<Set<string>>(new Set())

function toggleExpand(id: string) {
  const next = new Set(expanded.value)
  if (next.has(id)) next.delete(id)
  else next.add(id)
  expanded.value = next
}

async function loadDoc() {
  try {
    doc.value = await api.getDoc(props.docId)
  } catch (error) {
    ElMessage.error(errorMessage(error))
  }
}

async function loadChunks() {
  loading.value = true
  try {
    const page = await api.listChunks(props.docId, {
      page: query.page,
      size: query.size,
      keyword: query.keyword.trim() || undefined,
      only_enabled: query.only_enabled,
    })
    chunks.value = page.items
    total.value = page.total
  } catch (error) {
    ElMessage.error(errorMessage(error))
  } finally {
    loading.value = false
  }
}

async function loadPending() {
  try {
    const data = await api.pendingReindex(props.docId)
    pending.value = data.pending
  } catch {
    pending.value = 0
  }
}

onMounted(async () => {
  await Promise.all([loadDoc(), loadChunks(), loadPending()])
  await nextTick()
  const first = document.querySelector('.chunk.hit')
  first?.scrollIntoView({ block: 'center', behavior: 'smooth' })
})

watch(
  () => [query.page, query.size, query.only_enabled],
  loadChunks,
)

function search() {
  query.page = 1
  loadChunks()
}

async function toggle(chunk: Chunk) {
  const next = !chunk.enabled
  try {
    await api.updateChunk(chunk.id, { enabled: next })
    chunk.enabled = next
    ElMessage.success(next ? '已启用，可被检索命中' : '已禁用，检索不会再命中它')
  } catch (error) {
    ElMessage.error(errorMessage(error))
  }
}

function openEdit(chunk: Chunk) {
  editing.value = chunk
  draft.value = chunk.content
  dialogVisible.value = true
}

async function saveEdit() {
  if (!editing.value) return
  const text = draft.value.trim()
  if (!text) {
    ElMessage.warning('内容不能为空')
    return
  }
  saving.value = true
  try {
    const updated = await api.updateChunk(editing.value.id, { content: text })
    const index = chunks.value.findIndex((item) => item.id === updated.id)
    if (index >= 0) chunks.value[index] = updated
    dialogVisible.value = false
    // 改完不会自动重建向量，这一步必须让用户看见
    await loadPending()
    ElMessage.success('已保存。该块需要重建向量后才会按新内容被检索')
  } catch (error) {
    ElMessage.error(errorMessage(error))
  } finally {
    saving.value = false
  }
}

async function remove(chunk: Chunk) {
  try {
    await ElMessageBox.confirm('删除后该段内容不再参与检索，且无法恢复。', '确认删除文本块', {
      type: 'warning',
      confirmButtonText: '删除',
      cancelButtonText: '取消',
    })
  } catch {
    return
  }
  try {
    await api.deleteChunk(chunk.id)
    ElMessage.success('已删除')
    await loadChunks()
    await loadPending()
  } catch (error) {
    ElMessage.error(errorMessage(error))
  }
}
</script>

<template>
  <div class="page">
    <div class="page-head">
      <div class="head-left">
        <el-button :icon="ArrowLeft" text @click="router.push({ name: 'kb-detail', params: { kbId } })">
          返回
        </el-button>
        <div>
          <h2 class="page-title">文本块</h2>
          <div class="page-sub">{{ doc?.filename || '—' }} · 共 {{ total }} 块</div>
        </div>
      </div>
      <div class="actions">
        <el-button :icon="Refresh" :loading="loading" @click="loadChunks">刷新</el-button>
      </div>
    </div>

    <el-alert
      v-if="focusPage"
      type="info"
      show-icon
      closable
      class="banner"
      :title="`从引用跳转：第 ${focusPage} 页`"
      description="高亮的是该页切出来的文本块。想看原文请翻书核对，别只信模型转述。"
      @close="focusPage = null"
    />

    <el-alert
      v-if="pending > 0"
      type="warning"
      show-icon
      :closable="false"
      class="banner"
      title="有文本块改过但还没重建向量"
      :description="`${pending} 个块的内容已更新，但检索用的还是旧向量。重建功能在 M2 剩余项里，当前请以禁用代替修改。`"
    />

    <div class="panel filters">
      <el-input
        v-model="query.keyword"
        placeholder="按内容关键词过滤"
        :prefix-icon="Search"
        clearable
        style="max-width: 300px"
        @keyup.enter="search"
        @clear="search"
      />
      <el-checkbox v-model="query.only_enabled" label="只看启用中的" />
      <el-button @click="search">搜索</el-button>
    </div>

    <div v-loading="loading" class="list">
      <el-empty v-if="!loading && !chunks.length" description="没有匹配的文本块" />

      <article
        v-for="chunk in chunks"
        :key="chunk.id"
        class="chunk panel"
        :class="{ off: !chunk.enabled, hit: focusPage !== null && chunk.page_start === focusPage }"
      >
        <div class="chunk-side">
          <span class="seq mono">#{{ chunk.seq + 1 }}</span>
          <span class="page mono">p{{ chunk.page_start }}</span>
        </div>

        <div class="chunk-body">
          <p
            class="content"
            :class="{ clamped: !expanded.has(chunk.id) }"
            @click="toggleExpand(chunk.id)"
          >
            {{ chunk.content }}
          </p>
          <button
            v-if="chunk.content.length > 120"
            type="button"
            class="expand-btn"
            @click="toggleExpand(chunk.id)"
          >
            {{ expanded.has(chunk.id) ? '收起' : '展开全文' }}
          </button>
          <div class="chunk-foot">
            <span class="mono muted">{{ chunk.char_count }} 字</span>
            <span v-if="chunk.meta?.indexed === false" class="warn-flag">未向量化</span>
            <span v-if="!chunk.enabled" class="off-flag">已禁用</span>
          </div>
        </div>

        <div class="chunk-tools">
          <el-switch
            :model-value="chunk.enabled"
            size="small"
            inline-prompt
            active-text="启用"
            inactive-text="禁用"
            @change="toggle(chunk)"
          />
          <el-button text size="small" :icon="Edit" @click="openEdit(chunk)" />
          <el-button text size="small" type="danger" :icon="Delete" @click="remove(chunk)" />
        </div>
      </article>
    </div>

    <div class="pager">
      <el-pagination
        v-model:current-page="query.page"
        v-model:page-size="query.size"
        :total="total"
        :page-sizes="[20, 50, 100]"
        layout="total, sizes, prev, pager, next"
        background
      />
    </div>

    <el-dialog v-model="dialogVisible" title="编辑文本块" width="640px" :close-on-click-modal="false">
      <el-input v-model="draft" type="textarea" :rows="12" maxlength="8000" show-word-limit />
      <p class="dialog-hint muted">
        改完保存只会更新数据库里的内容，向量要重建才会跟着变。
      </p>
      <template #footer>
        <el-button @click="dialogVisible = false">取消</el-button>
        <el-button type="primary" :loading="saving" @click="saveEdit">保存</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.head-left {
  display: flex;
  align-items: center;
  gap: 10px;
}

.banner {
  margin-bottom: 14px;
}

.filters {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 14px;
  margin-bottom: 12px;
}

.list {
  display: flex;
  flex-direction: column;
  gap: 10px;
  min-height: 120px;
}

.chunk {
  display: flex;
  gap: 14px;
  padding: 14px 16px;
  transition: opacity 0.15s ease;
}

.chunk.off {
  opacity: 0.62;
}

.chunk.hit {
  border-color: var(--el-color-primary-light-5);
  box-shadow: 0 0 0 2px var(--el-color-primary-light-8);
}

.chunk-side {
  flex: 0 0 52px;
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding-top: 2px;
}

.seq {
  color: var(--el-color-primary);
  font-weight: 600;
}

.page {
  color: var(--app-text-weak);
}

.chunk-body {
  flex: 1;
  min-width: 0;
}

.content {
  margin: 0;
  font-size: 13px;
  line-height: 1.7;
  white-space: pre-wrap;
  word-break: break-word;
  cursor: pointer;
}

.content.clamped {
  display: -webkit-box;
  -webkit-line-clamp: 5;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.expand-btn {
  border: 0;
  background: transparent;
  color: var(--el-color-primary);
  font-size: 11.5px;
  padding: 4px 0 0;
  cursor: pointer;
}

.chunk-foot {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-top: 8px;
}

.warn-flag {
  font-size: 11px;
  color: var(--el-color-warning);
  background: var(--el-color-warning-light-9);
  border-radius: 4px;
  padding: 1px 5px;
}

.off-flag {
  font-size: 11px;
  color: var(--app-text-weak);
  background: #f1f2f6;
  border-radius: 4px;
  padding: 1px 5px;
}

.chunk-tools {
  flex: 0 0 auto;
  display: flex;
  align-items: center;
  gap: 4px;
}

.pager {
  display: flex;
  justify-content: flex-end;
  margin-top: 16px;
}

.dialog-hint {
  font-size: 12px;
  margin: 10px 0 0;
}
</style>
