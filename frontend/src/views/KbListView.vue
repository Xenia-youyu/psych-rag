<script setup lang="ts">
import { Collection, Delete, Edit, Plus, Refresh } from '@element-plus/icons-vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { onMounted, reactive, ref } from 'vue'
import { useRouter } from 'vue-router'

import { api } from '@/api'
import { errorMessage } from '@/api/client'
import type { Kb } from '@/api/types'

const router = useRouter()

const kbs = ref<Kb[]>([])
const loading = ref(false)
const dialogVisible = ref(false)
const saving = ref(false)
const editing = ref<Kb | null>(null)
const form = reactive({ name: '', description: '' })

async function load() {
  loading.value = true
  try {
    kbs.value = await api.listKbs()
  } catch (error) {
    ElMessage.error(errorMessage(error))
  } finally {
    loading.value = false
  }
}

onMounted(load)

function openCreate() {
  editing.value = null
  form.name = ''
  form.description = ''
  dialogVisible.value = true
}

function openEdit(kb: Kb) {
  editing.value = kb
  form.name = kb.name
  form.description = kb.description
  dialogVisible.value = true
}

async function save() {
  if (!form.name.trim()) {
    ElMessage.warning('知识库名称不能为空')
    return
  }
  saving.value = true
  try {
    if (editing.value) {
      await api.updateKb(editing.value.id, {
        name: form.name.trim(),
        description: form.description.trim(),
      })
      ElMessage.success('已保存')
    } else {
      await api.createKb({
        name: form.name.trim(),
        description: form.description.trim(),
      })
      ElMessage.success('知识库已创建')
    }
    dialogVisible.value = false
    await load()
  } catch (error) {
    ElMessage.error(errorMessage(error))
  } finally {
    saving.value = false
  }
}

async function remove(kb: Kb) {
  try {
    await ElMessageBox.confirm(
      `删除「${kb.name}」会同时移除它的 ${kb.doc_count} 个文档、${kb.chunk_count} 个文本块和全部向量，且无法恢复。`,
      '确认删除知识库',
      { type: 'warning', confirmButtonText: '删除', cancelButtonText: '取消' },
    )
  } catch {
    return
  }
  try {
    await api.deleteKb(kb.id)
    ElMessage.success('已删除')
    await load()
  } catch (error) {
    ElMessage.error(errorMessage(error))
  }
}

function open(kb: Kb) {
  router.push({ name: 'kb-detail', params: { kbId: kb.id } })
}
</script>

<template>
  <div class="page">
    <div class="page-head">
      <div>
        <h2 class="page-title">知识库</h2>
        <div class="page-sub">每个知识库是一组教材，检索只在所选库内进行</div>
      </div>
      <div class="actions">
        <el-button :icon="Refresh" :loading="loading" @click="load">刷新</el-button>
        <el-button type="primary" :icon="Plus" @click="openCreate">新建知识库</el-button>
      </div>
    </div>

    <div v-if="loading && !kbs.length" class="grid">
      <el-skeleton v-for="i in 3" :key="i" :rows="2" animated class="panel skeleton" />
    </div>

    <el-empty
      v-else-if="!kbs.length"
      description="还没有知识库。先建一个，再把教材传进去。"
    >
      <el-button type="primary" :icon="Plus" @click="openCreate">新建知识库</el-button>
    </el-empty>

    <div v-else class="grid">
      <article v-for="kb in kbs" :key="kb.id" class="card panel" @click="open(kb)">
        <div class="card-top">
          <div class="icon"><el-icon :size="18"><Collection /></el-icon></div>
          <div class="tools" @click.stop>
            <el-button text size="small" :icon="Edit" @click="openEdit(kb)" />
            <el-button text size="small" :icon="Delete" @click="remove(kb)" />
          </div>
        </div>

        <h3 class="name">{{ kb.name }}</h3>
        <p class="desc">{{ kb.description || '暂无描述' }}</p>

        <div class="stats">
          <div class="stat">
            <span class="num">{{ kb.doc_count }}</span>
            <span class="label">文档</span>
          </div>
          <div class="divider" />
          <div class="stat">
            <span class="num">{{ kb.chunk_count }}</span>
            <span class="label">文本块</span>
          </div>
        </div>
      </article>
    </div>

    <el-dialog
      v-model="dialogVisible"
      :title="editing ? '编辑知识库' : '新建知识库'"
      width="460px"
    >
      <el-form label-position="top">
        <el-form-item label="名称">
          <el-input v-model="form.name" maxlength="50" show-word-limit placeholder="例如：普通心理学" />
        </el-form-item>
        <el-form-item label="描述">
          <el-input
            v-model="form.description"
            type="textarea"
            :rows="3"
            maxlength="200"
            show-word-limit
            placeholder="这个库准备放哪些书（可选）"
          />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">取消</el-button>
        <el-button type="primary" :loading="saving" @click="save">保存</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.actions {
  display: flex;
  gap: 8px;
}

.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(264px, 1fr));
  gap: 14px;
}

.card {
  padding: 18px;
  cursor: pointer;
  transition: border-color 0.15s ease, transform 0.15s ease;
}

.card:hover {
  border-color: var(--el-color-primary-light-5);
  transform: translateY(-1px);
}

.card-top {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  margin-bottom: 12px;
}

.icon {
  width: 34px;
  height: 34px;
  border-radius: 9px;
  background: var(--el-color-primary-light-9);
  color: var(--el-color-primary);
  display: flex;
  align-items: center;
  justify-content: center;
}

.tools {
  display: flex;
  opacity: 0;
  transition: opacity 0.15s ease;
}

.card:hover .tools {
  opacity: 1;
}

.name {
  font-size: 15px;
  font-weight: 600;
  margin: 0 0 6px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.desc {
  font-size: 12.5px;
  color: var(--app-text-weak);
  margin: 0 0 14px;
  min-height: 34px;
  line-height: 1.45;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.stats {
  display: flex;
  align-items: center;
  gap: 14px;
  padding-top: 12px;
  border-top: 1px solid var(--app-border);
}

.stat {
  display: flex;
  align-items: baseline;
  gap: 5px;
}

.num {
  font-size: 15px;
  font-weight: 600;
}

.label {
  font-size: 11.5px;
  color: var(--app-text-weak);
}

.divider {
  width: 1px;
  height: 14px;
  background: var(--app-border);
}

.skeleton {
  padding: 18px;
}
</style>
