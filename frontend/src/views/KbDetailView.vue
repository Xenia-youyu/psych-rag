<script setup lang="ts">
import { ArrowLeft, Delete, Document, Refresh, Upload } from '@element-plus/icons-vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { computed, onBeforeUnmount, onMounted, reactive, ref } from 'vue'
import { useRouter } from 'vue-router'

import { api } from '@/api'
import { errorMessage } from '@/api/client'
import type { Doc, DocStatus, Kb, KbStats } from '@/api/types'

const props = defineProps<{ kbId: string }>()
const router = useRouter()

const kb = ref<Kb | null>(null)
const stats = ref<KbStats | null>(null)
const docs = ref<Doc[]>([])
const loading = ref(false)
const uploading = ref(false)

const dialogVisible = ref(false)
const fileInput = ref<HTMLInputElement | null>(null)
const params = reactive({
  parent_size: 1200,
  child_size: 320,
  overlap_ratio: 0.1,
})

interface QueueItem {
  file: File
  percent: number
  state: 'wait' | 'uploading' | 'done' | 'error'
  message: string
}
const queue = ref<QueueItem[]>([])

const PROCESSING: DocStatus[] = ['uploaded', 'parsing', 'chunking', 'embedding']

const STATUS_META: Record<DocStatus, { text: string; type: 'info' | 'warning' | 'success' | 'danger' }> = {
  uploaded: { text: '排队中', type: 'info' },
  parsing: { text: '解析中', type: 'warning' },
  chunking: { text: '切块中', type: 'warning' },
  embedding: { text: '向量化', type: 'warning' },
  done: { text: '已完成', type: 'success' },
  failed: { text: '失败', type: 'danger' },
}

const busy = computed(() => docs.value.some((doc) => PROCESSING.includes(doc.status)))

// ---------------------------------------------------------------- 轮询
// 一本 300 页的书要跑十几分钟。固定 2 秒轮询会白打几百个请求，
// 所以状态连续三次没变化就退避到 5 秒；全部结束立刻停。
let timer: number | null = null
let unchanged = 0

function signature() {
  return docs.value.map((doc) => `${doc.id}:${doc.status}`).join('|')
}

function stopPolling() {
  if (timer !== null) {
    window.clearTimeout(timer)
    timer = null
  }
}

function schedulePolling() {
  stopPolling()
  if (!busy.value) return

  const delay = unchanged >= 3 ? 5000 : 2000
  timer = window.setTimeout(async () => {
    const before = signature()
    await loadDocs(true)
    unchanged = signature() === before ? unchanged + 1 : 0
    schedulePolling()
  }, delay)
}

// ---------------------------------------------------------------- 数据
async function loadAll() {
  loading.value = true
  try {
    const [kbData, statsData, docList] = await Promise.all([
      api.getKb(props.kbId),
      api.kbStats(props.kbId),
      api.listDocs(props.kbId),
    ])
    kb.value = kbData
    stats.value = statsData
    docs.value = docList
    schedulePolling()
  } catch (error) {
    ElMessage.error(errorMessage(error))
  } finally {
    loading.value = false
  }
}

async function loadDocs(silent = false) {
  try {
    docs.value = await api.listDocs(props.kbId)
    schedulePolling()
  } catch (error) {
    if (!silent) ElMessage.error(errorMessage(error))
  }
}

async function refreshStats() {
  try {
    stats.value = await api.kbStats(props.kbId)
  } catch {
    // 统计失败不该打断主流程
  }
}

onMounted(loadAll)
onBeforeUnmount(stopPolling)

// ---------------------------------------------------------------- 上传
function pickFiles(event: Event) {
  const input = event.target as HTMLInputElement
  const files = Array.from(input.files || [])
  if (!files.length) return
  queue.value = files.map((file) => ({ file, percent: 0, state: 'wait', message: '' }))
  dialogVisible.value = true
  input.value = ''
}

async function startUpload() {
  if (!queue.value.length) return
  uploading.value = true
  try {
    for (const item of queue.value) {
      if (item.state === 'done') continue
      item.state = 'uploading'
      try {
        const doc = await api.uploadDoc(
          props.kbId,
          item.file,
          {
            chunk_size: 800,
            overlap_ratio: params.overlap_ratio,
            hybrid: true,
            mode: 'parent_child',
            parent_size: params.parent_size,
            child_size: params.child_size,
          },
          (percent) => (item.percent = percent),
        )
        item.state = 'done'
        item.percent = 100
        // 去重命中不是失败：后端把库里已有的那条原样返回了
        item.message = doc.duplicated ? '该文件已在库中，本次未重复解析' : '已提交，后台解析中'
      } catch (error) {
        item.state = 'error'
        item.message = errorMessage(error)
      }
    }
    await loadDocs(true)
    await refreshStats()
    unchanged = 0
    schedulePolling()
  } finally {
    uploading.value = false
  }
}

function closeDialog() {
  if (uploading.value) return
  dialogVisible.value = false
  queue.value = []
}

// ---------------------------------------------------------------- 操作
async function remove(doc: Doc) {
  try {
    await ElMessageBox.confirm(
      `删除「${doc.filename}」会同时清掉它的文本块和向量，且无法恢复。`,
      '确认删除文档',
      { type: 'warning', confirmButtonText: '删除', cancelButtonText: '取消' },
    )
  } catch {
    return
  }
  try {
    await api.deleteDoc(doc.id)
    ElMessage.success('已删除')
    await loadDocs(true)
    await refreshStats()
  } catch (error) {
    ElMessage.error(errorMessage(error))
  }
}

function openChunks(doc: Doc) {
  router.push({ name: 'chunks', params: { kbId: props.kbId, docId: doc.id } })
}

function formatSize(bytes: number) {
  if (!bytes) return '-'
  const mb = bytes / 1024 / 1024
  return mb >= 1 ? `${mb.toFixed(1)} MB` : `${Math.round(bytes / 1024)} KB`
}
</script>

<template>
  <div class="page">
    <div class="page-head">
      <div class="head-left">
        <el-button :icon="ArrowLeft" text @click="router.push({ name: 'kb-list' })">返回</el-button>
        <div>
          <h2 class="page-title">{{ kb?.name || '知识库' }}</h2>
          <div class="page-sub">{{ kb?.description || '暂无描述' }}</div>
        </div>
      </div>
      <div class="actions">
        <el-button :icon="Refresh" :loading="loading" @click="loadAll">刷新</el-button>
        <el-button type="primary" :icon="Upload" @click="fileInput?.click()">上传文档</el-button>
        <input
          ref="fileInput"
          type="file"
          multiple
          accept=".pdf,.docx,.doc,.txt,.md,.png,.jpg,.jpeg"
          hidden
          @change="pickFiles"
        />
      </div>
    </div>

    <div class="stats-row">
      <div class="stat panel">
        <div class="stat-num">{{ stats?.documents ?? 0 }}</div>
        <div class="stat-label">文档</div>
      </div>
      <div class="stat panel">
        <div class="stat-num">{{ stats?.chunks ?? 0 }}</div>
        <div class="stat-label">文本块</div>
      </div>
      <div class="stat panel">
        <div class="stat-num">{{ stats?.enabled_chunks ?? 0 }}</div>
        <div class="stat-label">启用中</div>
      </div>
      <div class="stat panel" :class="{ danger: (stats?.failed_documents ?? 0) > 0 }">
        <div class="stat-num">{{ stats?.failed_documents ?? 0 }}</div>
        <div class="stat-label">解析失败</div>
      </div>
    </div>

    <div class="panel table-wrap">
      <el-table :data="docs" v-loading="loading" empty-text="还没有文档，上传一本教材开始">
        <el-table-column label="文件" min-width="260">
          <template #default="{ row }">
            <div class="file-cell">
              <el-icon class="file-icon"><Document /></el-icon>
              <div class="file-text">
                <div class="file-name">{{ row.filename }}</div>
                <div class="file-meta mono">
                  {{ row.file_type }} · {{ formatSize(row.size_bytes) }} · {{ row.page_count }} 页
                </div>
              </div>
            </div>
          </template>
        </el-table-column>

        <el-table-column label="状态" width="150">
          <template #default="{ row }">
            <div class="status-cell">
              <el-tag :type="STATUS_META[row.status as DocStatus].type" size="small" effect="light">
                {{ STATUS_META[row.status as DocStatus].text }}
              </el-tag>
              <el-tooltip v-if="row.status === 'failed'" :content="row.error_msg" placement="top">
                <span class="err-dot">!</span>
              </el-tooltip>
            </div>
            <div v-if="PROCESSING.includes(row.status)" class="stage mono">
              {{ row.last_stage }}
            </div>
          </template>
        </el-table-column>

        <el-table-column label="创建时间" width="160">
          <template #default="{ row }">
            <span class="mono muted">{{ (row.created_at || '').replace('T', ' ').slice(0, 16) }}</span>
          </template>
        </el-table-column>

        <el-table-column label="操作" width="160" align="right">
          <template #default="{ row }">
            <el-button text size="small" :disabled="row.status !== 'done'" @click="openChunks(row)">
              文本块
            </el-button>
            <el-button text size="small" type="danger" :icon="Delete" @click="remove(row)" />
          </template>
        </el-table-column>
      </el-table>
    </div>

    <p v-if="busy" class="polling-hint muted">
      有文档正在后台解析，页面每 {{ unchanged >= 3 ? 5 : 2 }} 秒自动刷新一次
    </p>

    <el-dialog
      v-model="dialogVisible"
      title="上传文档"
      width="560px"
      :close-on-click-modal="false"
      @close="closeDialog"
    >
      <div class="queue">
        <div v-for="(item, index) in queue" :key="index" class="queue-item">
          <div class="queue-head">
            <span class="queue-name">{{ item.file.name }}</span>
            <span class="mono muted">{{ formatSize(item.file.size) }}</span>
          </div>
          <el-progress
            :percentage="item.percent"
            :status="item.state === 'error' ? 'exception' : item.state === 'done' ? 'success' : undefined"
            :stroke-width="6"
          />
          <div v-if="item.message" class="queue-msg" :class="{ bad: item.state === 'error' }">
            {{ item.message }}
          </div>
        </div>
      </div>

      <div class="params">
        <div class="params-title">切块参数</div>
        <p class="params-hint muted">
          只影响本次入库。父子块模式下，检索按 {{ params.child_size }} 字的子块匹配，
          命中后回填 {{ params.parent_size }} 字的父块给模型。
        </p>
        <div class="param-row">
          <label>父块大小</label>
          <el-input-number v-model="params.parent_size" :min="800" :max="2000" :step="100" size="small" />
          <span class="muted mono">字</span>
        </div>
        <div class="param-row">
          <label>子块大小</label>
          <el-input-number v-model="params.child_size" :min="150" :max="600" :step="20" size="small" />
          <span class="muted mono">字</span>
        </div>
        <div class="param-row">
          <label>重叠比例</label>
          <el-input-number v-model="params.overlap_ratio" :min="0" :max="0.3" :step="0.05" size="small" />
          <span class="muted mono">0 ~ 0.3</span>
        </div>
      </div>

      <template #footer>
        <el-button :disabled="uploading" @click="closeDialog">关闭</el-button>
        <el-button type="primary" :loading="uploading" @click="startUpload">
          {{ uploading ? '上传中…' : '开始上传' }}
        </el-button>
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

.actions {
  display: flex;
  gap: 8px;
}

.stats-row {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
  margin-bottom: 14px;
}

.stat {
  padding: 14px 16px;
}

.stat.danger .stat-num {
  color: var(--el-color-danger);
}

.stat-num {
  font-size: 21px;
  font-weight: 600;
  line-height: 1.1;
}

.stat-label {
  font-size: 12px;
  color: var(--app-text-weak);
  margin-top: 3px;
}

.table-wrap {
  padding: 6px 6px 2px;
  overflow: hidden;
}

.file-cell {
  display: flex;
  align-items: center;
  gap: 9px;
}

.file-icon {
  color: var(--el-color-primary);
  flex: 0 0 auto;
}

.file-text {
  min-width: 0;
}

.file-name {
  font-size: 13.5px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.file-meta {
  color: var(--app-text-weak);
  margin-top: 2px;
}

.status-cell {
  display: flex;
  align-items: center;
  gap: 6px;
}

.err-dot {
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: var(--el-color-danger);
  color: #fff;
  font-size: 11px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  cursor: help;
}

.stage {
  color: var(--app-text-weak);
  margin-top: 4px;
}

.polling-hint {
  font-size: 12px;
  margin-top: 10px;
}

.queue {
  display: flex;
  flex-direction: column;
  gap: 14px;
  margin-bottom: 18px;
}

.queue-head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: 6px;
}

.queue-name {
  font-size: 13px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.queue-msg {
  font-size: 11.5px;
  color: var(--el-color-success);
  margin-top: 5px;
}

.queue-msg.bad {
  color: var(--el-color-danger);
}

.params {
  background: #fafbfd;
  border: 1px solid var(--app-border);
  border-radius: 10px;
  padding: 14px 16px;
}

.params-title {
  font-size: 13px;
  font-weight: 600;
  margin-bottom: 4px;
}

.params-hint {
  font-size: 11.5px;
  margin: 0 0 12px;
  line-height: 1.55;
}

.param-row {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 8px;
}

.param-row label {
  width: 70px;
  font-size: 12.5px;
  color: var(--app-text-weak);
}

.param-row:last-child {
  margin-bottom: 0;
}
</style>
