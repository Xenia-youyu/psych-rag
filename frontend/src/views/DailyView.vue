<script setup lang="ts">
import { ArrowRight, Refresh, Search } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'

import { api } from '@/api'
import { errorMessage } from '@/api/client'
import type { EffectBrief, EffectDetail } from '@/api/types'

const router = useRouter()

const effect = ref<EffectDetail | null>(null)
const variation = ref(0)
const today = ref('')
const total = ref(0)
const loading = ref(false)

const catalog = ref<EffectBrief[]>([])
const categories = ref<string[]>([])
const activeCategory = ref('')
const keyword = ref('')
const showCatalog = ref(false)

const filtered = computed(() => {
  const kw = keyword.value.trim()
  return catalog.value.filter((item) => {
    if (activeCategory.value && item.category !== activeCategory.value) return false
    if (!kw) return true
    return (
      item.name.includes(kw) || item.name_en.toLowerCase().includes(kw.toLowerCase()) || item.summary.includes(kw)
    )
  })
})

async function loadDaily() {
  loading.value = true
  try {
    const data = await api.dailyEffects(variation.value, 1)
    effect.value = data.items[0] || null
    today.value = data.date
    total.value = data.total
  } catch (error) {
    ElMessage.error(errorMessage(error))
  } finally {
    loading.value = false
  }
}

async function loadCatalog() {
  try {
    const data = await api.listEffects()
    catalog.value = data.items
    categories.value = data.categories
  } catch (error) {
    ElMessage.error(errorMessage(error))
  }
}

onMounted(async () => {
  await Promise.all([loadDaily(), loadCatalog()])
})

function next() {
  variation.value = (variation.value + 1) % 99
  loadDaily()
}

function ask() {
  if (!effect.value) return
  router.push({
    path: '/',
    query: { q: `请解释一下「${effect.value.name}」，它和日常生活有什么关系？` },
  })
}

async function open(slug: string) {
  try {
    effect.value = await api.getEffect(slug)
    variation.value = 0
    showCatalog.value = false
    window.scrollTo({ top: 0, behavior: 'smooth' })
  } catch (error) {
    ElMessage.error(errorMessage(error))
  }
}

function formatDate(value: string) {
  if (!value) return ''
  const [y, m, d] = value.split('-')
  return `${y} 年 ${Number(m)} 月 ${Number(d)} 日`
}
</script>

<template>
  <div class="page">
    <header class="head">
      <div>
        <h1>每日知识</h1>
        <p class="sub">一天一个心理学效应 · 与你的知识库无关，人人相同</p>
      </div>
      <div class="head-actions">
        <span v-if="today" class="date">{{ formatDate(today) }}</span>
        <el-button :icon="Refresh" @click="next">换一个</el-button>
      </div>
    </header>

    <div v-if="effect" v-loading="loading" class="feature">
      <div class="feature-top">
        <div class="tags">
          <span class="tag-cat">{{ effect.category }}</span>
          <span v-if="effect.founder" class="tag-meta">
            {{ effect.founder }}<template v-if="effect.year"> · {{ effect.year }}</template>
          </span>
        </div>
        <div class="index">No.{{ (catalog.findIndex((i) => i.slug === effect!.slug) + 1) || 1 }} / {{ total }}</div>
      </div>

      <h2 class="name">{{ effect.name }}</h2>
      <p v-if="effect.name_en" class="name-en">{{ effect.name_en }}</p>

      <p class="summary">{{ effect.summary }}</p>

      <div class="block">
        <div class="block-title">它是什么</div>
        <p class="block-text">{{ effect.detail }}</p>
      </div>

      <div v-if="effect.experiment" class="block is-accent">
        <div class="block-title">经典研究</div>
        <p class="block-text">{{ effect.experiment }}</p>
      </div>

      <div v-if="effect.example" class="block">
        <div class="block-title">日常场景</div>
        <p class="block-text">{{ effect.example }}</p>
      </div>

      <div v-if="effect.related.length" class="related">
        <span class="related-label">相关效应</span>
        <button
          v-for="item in effect.related"
          :key="item.slug"
          type="button"
          class="related-item"
          @click="open(item.slug)"
        >
          {{ item.name }}
        </button>
      </div>

      <div class="feature-foot">
        <el-button type="primary" @click="ask">
          就这个效应问一问
          <el-icon class="ml"><ArrowRight /></el-icon>
        </el-button>
        <span v-if="effect.source" class="source">参考：{{ effect.source }}</span>
      </div>
    </div>

    <div v-else v-loading="loading" class="loading-holder" />

    <section class="catalog">
      <button type="button" class="catalog-toggle" @click="showCatalog = !showCatalog">
        {{ showCatalog ? '收起全部效应' : `浏览全部 ${catalog.length} 个心理学效应` }}
      </button>

      <div v-if="showCatalog" class="catalog-body">
        <div class="catalog-tools">
          <el-input v-model="keyword" placeholder="搜索效应名称" clearable :prefix-icon="Search" />
          <div class="cats">
            <button
              type="button"
              class="cat"
              :class="{ 'is-on': !activeCategory }"
              @click="activeCategory = ''"
            >
              全部
            </button>
            <button
              v-for="cat in categories"
              :key="cat"
              type="button"
              class="cat"
              :class="{ 'is-on': activeCategory === cat }"
              @click="activeCategory = cat"
            >
              {{ cat }}
            </button>
          </div>
        </div>

        <div class="grid">
          <button
            v-for="item in filtered"
            :key="item.slug"
            type="button"
            class="tile"
            @click="open(item.slug)"
          >
            <div class="tile-name">{{ item.name }}</div>
            <div class="tile-en">{{ item.name_en }}</div>
            <div class="tile-summary">{{ item.summary }}</div>
          </button>
          <p v-if="!filtered.length" class="tile-empty">没有匹配的效应</p>
        </div>
      </div>
    </section>
  </div>
</template>

<style scoped>
.page {
  max-width: 820px;
  margin: 0 auto;
  padding: 30px 32px 70px;
}

.head {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 20px;
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

.head-actions {
  display: flex;
  align-items: center;
  gap: 10px;
  flex: 0 0 auto;
}

.date {
  font-size: 12px;
  color: var(--app-text-weak);
}

.feature {
  background: #fff;
  border: 1px solid var(--app-border);
  border-radius: 16px;
  padding: 26px 30px 24px;
  box-shadow: 0 6px 24px rgba(30, 40, 90, 0.05);
  min-height: 260px;
}

.feature-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.tags {
  display: flex;
  align-items: center;
  gap: 8px;
}

.tag-cat {
  font-size: 11.5px;
  color: var(--el-color-primary-dark-2);
  background: var(--el-color-primary-light-9);
  border-radius: 999px;
  padding: 2px 10px;
}

.tag-meta {
  font-size: 11.5px;
  color: var(--app-text-weak);
}

.index {
  font-size: 11.5px;
  color: var(--app-text-weak);
  letter-spacing: 0.4px;
}

.name {
  font-size: 27px;
  font-weight: 600;
  margin: 14px 0 2px;
  letter-spacing: 0.3px;
}

.name-en {
  margin: 0 0 14px;
  font-size: 12.5px;
  color: var(--app-text-weak);
  letter-spacing: 0.5px;
}

.summary {
  margin: 0 0 20px;
  font-size: 15px;
  line-height: 1.85;
  color: var(--app-text);
  font-weight: 500;
}

.block {
  margin-bottom: 16px;
}

.block.is-accent {
  background: #f8f9ff;
  border-radius: 10px;
  padding: 13px 15px;
}

.block-title {
  font-size: 12px;
  font-weight: 600;
  color: var(--el-color-primary-dark-2);
  letter-spacing: 0.4px;
  margin-bottom: 6px;
}

.block-text {
  margin: 0;
  font-size: 13.5px;
  line-height: 1.95;
  color: #3d4458;
}

.related {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
  padding-top: 6px;
}

.related-label {
  font-size: 12px;
  color: var(--app-text-weak);
}

.related-item {
  border: 1px solid var(--app-border);
  background: #fff;
  border-radius: 999px;
  padding: 3px 11px;
  font-size: 12px;
  color: var(--app-text);
  cursor: pointer;
}

.related-item:hover {
  border-color: var(--el-color-primary-light-5);
  color: var(--el-color-primary-dark-2);
  background: var(--el-color-primary-light-9);
}

.feature-foot {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
  margin-top: 20px;
  padding-top: 16px;
  border-top: 1px dashed var(--app-border);
}

.source {
  font-size: 11.5px;
  color: var(--app-text-weak);
  text-align: right;
  line-height: 1.6;
}

.ml {
  margin-left: 4px;
}

.loading-holder {
  height: 260px;
}

/* ------------------------------------------------------------ 目录 */
.catalog {
  margin-top: 24px;
}

.catalog-toggle {
  width: 100%;
  border: 1px dashed var(--app-border);
  background: #fff;
  border-radius: 11px;
  padding: 11px;
  font-size: 13px;
  color: var(--app-text-weak);
  cursor: pointer;
}

.catalog-toggle:hover {
  border-color: var(--el-color-primary-light-5);
  color: var(--el-color-primary-dark-2);
}

.catalog-body {
  margin-top: 16px;
}

.catalog-tools {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-bottom: 14px;
}

.cats {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
}

.cat {
  border: 1px solid var(--app-border);
  background: #fff;
  border-radius: 999px;
  padding: 4px 12px;
  font-size: 12px;
  color: var(--app-text-weak);
  cursor: pointer;
}

.cat.is-on {
  background: var(--el-color-primary);
  border-color: var(--el-color-primary);
  color: #fff;
}

.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(228px, 1fr));
  gap: 10px;
}

.tile {
  text-align: left;
  background: #fff;
  border: 1px solid var(--app-border);
  border-radius: 11px;
  padding: 13px 14px;
  cursor: pointer;
  transition: all 0.15s ease;
}

.tile:hover {
  border-color: var(--el-color-primary-light-5);
  box-shadow: 0 4px 14px rgba(30, 40, 90, 0.06);
}

.tile-name {
  font-size: 13.5px;
  font-weight: 500;
}

.tile-en {
  font-size: 11px;
  color: var(--app-text-weak);
  margin: 2px 0 6px;
}

.tile-summary {
  font-size: 12px;
  line-height: 1.65;
  color: var(--app-text-weak);
  display: -webkit-box;
  -webkit-line-clamp: 2;
  line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.tile-empty {
  grid-column: 1 / -1;
  text-align: center;
  font-size: 12.5px;
  color: var(--app-text-weak);
  padding: 24px 0;
}
</style>
