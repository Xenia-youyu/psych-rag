<script setup lang="ts">
import { ChatLineRound, Document, Reading, Refresh, Right } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import { computed, onMounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'

import { api } from '@/api'
import { errorMessage } from '@/api/client'
import type { Kb, RecommendDetail, RecommendItem } from '@/api/types'

const router = useRouter()

const kbs = ref<Kb[]>([])
const kbId = ref('')
const variation = ref(0)
const date = ref('')
const candidates = ref(0)

const items = ref<RecommendItem[]>([])
const loadingList = ref(false)

const detail = ref<RecommendDetail | null>(null)
const loadingDetail = ref(false)

const empty = computed(() => !loadingList.value && items.value.length === 0)

async function loadKbs() {
  try {
    kbs.value = await api.listKbs()
    const withDocs = kbs.value.find((kb) => kb.doc_count > 0) || kbs.value[0]
    if (withDocs) kbId.value = withDocs.id
  } catch (error) {
    ElMessage.error(errorMessage(error))
  }
}

async function loadDaily() {
  if (!kbId.value) return
  loadingList.value = true
  detail.value = null
  try {
    const data = await api.recommendDaily(kbId.value, variation.value)
    items.value = data.items
    date.value = data.date
    candidates.value = data.total_candidates
    if (data.items.length) await openDetail(data.items[0].chunk_id)
  } catch (error) {
    ElMessage.error(errorMessage(error))
    items.value = []
  } finally {
    loadingList.value = false
  }
}

async function openDetail(chunkId: string) {
  loadingDetail.value = true
  try {
    detail.value = await api.recommendDetail(chunkId)
  } catch (error) {
    ElMessage.error(errorMessage(error))
  } finally {
    loadingDetail.value = false
  }
}

function nextBatch() {
  // 同一天换一批：后端给 seed 加偏移，结果仍然固定可复现
  variation.value += 1
  loadDaily()
}

function askAbout(title: string) {
  // 标题可能带首尾省略号（表示节选），原样带进问题里读起来很怪
  const clean = title.replace(/^…/, '').replace(/…$/, '')
  router.push({
    name: 'chat',
    query: { q: `请解释这段内容，并说明它的上下文：${clean}`, kb: kbId.value },
  })
}

onMounted(loadKbs)
watch(kbId, loadDaily)

function formatDate(value: string) {
  if (!value) return ''
  const [year, month, day] = value.split('-')
  return `${year} 年 ${Number(month)} 月 ${Number(day)} 日`
}
</script>

<template>
  <div class="page">
    <div class="page-head">
      <div>
        <h2 class="page-title">每日知识</h2>
        <div class="page-sub">
          {{ formatDate(date) || '今天' }}从知识库里挑出的内容，当天固定不变
          <span v-if="candidates" class="muted">· 候选 {{ candidates }} 条</span>
        </div>
      </div>
      <div class="actions">
        <el-select v-model="kbId" placeholder="选择知识库" style="width: 200px">
          <el-option v-for="kb in kbs" :key="kb.id" :label="kb.name" :value="kb.id" />
        </el-select>
        <el-button :icon="Refresh" :loading="loadingList" @click="loadDaily">刷新</el-button>
        <el-button :disabled="!items.length" @click="nextBatch">换一批</el-button>
      </div>
    </div>

    <el-empty
      v-if="!kbs.length"
      description="还没有知识库。先去「知识库」传几本教材，这里每天就会有内容了。"
    />

    <div v-else class="layout">
      <aside class="column">
        <div class="column-head">今日 {{ items.length }} 条</div>

        <el-skeleton v-if="loadingList" :rows="4" animated class="panel skeleton" />

        <template v-else>
          <el-empty v-if="empty" description="这个知识库还没有可推荐的内容" :image-size="70" />

          <button
            v-for="item in items"
            :key="item.chunk_id"
            type="button"
            class="card panel"
            :class="{ 'is-on': detail?.chunk_id === item.chunk_id }"
            @click="openDetail(item.chunk_id)"
          >
            <div class="card-title">{{ item.title }}</div>
            <p class="card-preview">{{ item.preview }}</p>
            <div class="card-meta">
              <el-icon :size="12"><Document /></el-icon>
              <span class="mono">{{ item.doc_title }}</span>
              <span class="dot" />
              <span class="mono">第 {{ item.page }} 页</span>
            </div>
          </button>
        </template>
      </aside>

      <section v-loading="loadingDetail" class="detail panel">
        <div v-if="!detail" class="detail-empty muted">
          <el-icon :size="26"><Reading /></el-icon>
          <p>选左边一条开始看</p>
        </div>

        <template v-else>
          <div class="detail-head">
            <h3>{{ detail.title }}</h3>
            <div class="detail-meta">
              <span>{{ detail.doc_title }}</span>
              <span class="dot" />
              <span>第 {{ detail.page }} 页</span>
              <span class="dot" />
              <span>{{ detail.char_count }} 字</span>
              <el-tag v-if="detail.is_full_parent" size="small" type="info" effect="plain" class="tag">
                所在章节全文
              </el-tag>
            </div>
          </div>

          <article class="prose">{{ detail.content }}</article>

          <div class="cta">
            <div class="cta-text">
              <strong>还想再深一点？</strong>
              <span class="muted">带着这段内容开一个新对话，让 AI 结合全书回答</span>
            </div>
            <el-button type="primary" :icon="ChatLineRound" @click="askAbout(detail.title)">
              就此提问
            </el-button>
          </div>

          <div v-if="detail.related.length" class="related">
            <div class="related-head">
              相关内容
              <span class="muted">· 按语义相近挑的，点开可换到那一条</span>
            </div>
            <div class="related-list">
              <button
                v-for="item in detail.related"
                :key="item.chunk_id"
                type="button"
                class="related-card"
                @click="openDetail(item.chunk_id)"
              >
                <div class="related-meta">
                  <span class="mono">{{ item.doc_title }}</span>
                  <span class="dot" />
                  <span class="mono">第 {{ item.page }} 页</span>
                </div>
                <p class="related-preview">{{ item.preview }}</p>
                <el-icon class="related-arrow" :size="13"><Right /></el-icon>
              </button>
            </div>
          </div>
        </template>
      </section>
    </div>
  </div>
</template>

<style scoped>
.actions {
  display: flex;
  gap: 8px;
}

.layout {
  display: grid;
  grid-template-columns: 336px minmax(0, 1fr);
  gap: 16px;
  align-items: start;
}

.column {
  position: sticky;
  top: 24px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.column-head {
  font-size: 12px;
  color: var(--app-text-weak);
  letter-spacing: 0.5px;
}

.card {
  text-align: left;
  padding: 14px 16px;
  cursor: pointer;
  border: 1px solid var(--app-border);
  transition: border-color 0.15s ease, background 0.15s ease;
  font: inherit;
}

.card:hover {
  border-color: var(--el-color-primary-light-5);
}

.card.is-on {
  border-color: var(--el-color-primary-light-3);
  background: var(--el-color-primary-light-9);
}

.card-title {
  font-size: 13.5px;
  font-weight: 600;
  line-height: 1.45;
  margin-bottom: 6px;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.card-preview {
  margin: 0 0 10px;
  font-size: 12px;
  line-height: 1.6;
  color: var(--app-text-weak);
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.card-meta {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 11px;
  color: var(--app-text-weak);
}

.dot {
  width: 3px;
  height: 3px;
  border-radius: 50%;
  background: var(--el-border-color);
  display: inline-block;
}

.detail {
  padding: 22px 26px 26px;
  min-height: 420px;
}

.detail-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  height: 380px;
}

.detail-head h3 {
  font-size: 17px;
  font-weight: 600;
  margin: 0 0 8px;
  line-height: 1.45;
}

.detail-meta {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 6px;
  font-size: 12px;
  color: var(--app-text-weak);
  padding-bottom: 14px;
  border-bottom: 1px solid var(--app-border);
}

.tag {
  margin-left: 4px;
}

.prose {
  font-size: 14px;
  line-height: 1.9;
  color: #2b3038;
  white-space: pre-wrap;
  word-break: break-word;
  margin: 18px 0 22px;
  max-height: 46vh;
  overflow: auto;
  padding-right: 6px;
}

.cta {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 14px 16px;
  border-radius: 10px;
  background: var(--el-color-primary-light-9);
  border: 1px solid var(--el-color-primary-light-8);
}

.cta-text {
  display: flex;
  flex-direction: column;
  gap: 3px;
  font-size: 13px;
}

.cta-text .muted {
  font-size: 11.5px;
}

.related {
  margin-top: 22px;
}

.related-head {
  font-size: 12.5px;
  font-weight: 600;
  margin-bottom: 10px;
}

.related-head .muted {
  font-weight: 400;
  font-size: 11.5px;
}

.related-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.related-card {
  position: relative;
  text-align: left;
  padding: 11px 34px 11px 13px;
  border: 1px solid var(--app-border);
  border-radius: 9px;
  background: #fafbfd;
  cursor: pointer;
  font: inherit;
}

.related-card:hover {
  border-color: var(--el-color-primary-light-5);
  background: #fff;
}

.related-meta {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 11px;
  color: var(--app-text-weak);
  margin-bottom: 5px;
}

.related-preview {
  margin: 0;
  font-size: 12.5px;
  line-height: 1.6;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.related-arrow {
  position: absolute;
  right: 12px;
  top: 50%;
  transform: translateY(-50%);
  color: var(--app-text-weak);
}

.skeleton {
  padding: 16px;
}

@media (max-width: 960px) {
  .layout {
    grid-template-columns: 1fr;
  }

  .column {
    position: static;
  }
}
</style>
