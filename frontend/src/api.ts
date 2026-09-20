const API_BASE = '/api/v1'

// ── 访问令牌 ──────────────────────────────────────────────────
// 后端 AUTH_SECRET 留空时不启用鉴权，此处没有令牌也能正常请求；
// 一旦后端配置了 AUTH_SECRET，就必须在「设置」页填入同样的值。

const AUTH_TOKEN_KEY = 'laborrag_auth_token'

export function getAuthToken(): string {
  return localStorage.getItem(AUTH_TOKEN_KEY) ?? ''
}

export function setAuthToken(token: string): void {
  const trimmed = token.trim()
  if (trimmed) localStorage.setItem(AUTH_TOKEN_KEY, trimmed)
  else localStorage.removeItem(AUTH_TOKEN_KEY)
}

function authHeaders(): Record<string, string> {
  const token = getAuthToken()
  return token ? { Authorization: `Bearer ${token}` } : {}
}

/**
 * 统一请求入口：注入 Bearer 凭证，并合并调用方自定义头。
 * 鉴权逻辑只在这里存在一处，新增接口不会漏带头。
 */
async function apiFetch(path: string, init: RequestInit = {}): Promise<Response> {
  return fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { ...authHeaders(), ...((init.headers as Record<string, string>) || {}) },
  })
}

/**
 * 统一错误消息：401 单独提示，避免与业务错误混为一谈。
 * 后端未启用鉴权却填了令牌，或令牌与 AUTH_SECRET 不一致，都会落到这里。
 */
async function readError(res: Response, fallback: string): Promise<string> {
  if (res.status === 401) {
    return '未授权：请在「设置」页填入与后端 AUTH_SECRET 一致的访问令牌'
  }
  const err = await res.json().catch(() => ({ detail: res.statusText }))
  return err.detail || fallback
}

export interface ChatHistoryItem {
  role: 'user' | 'assistant'
  content: string
}

export interface ChatRequest {
  question: string
  conversation_id?: string
  history?: ChatHistoryItem[]
}

export interface SourceDocument {
  content: string
  source: string
  score: number
  page?: number
}

export interface ChatResponse {
  answer: string
  sources: SourceDocument[]
  conversation_id: string
  rag_steps: string[]
  rewritten_question: string
  confidence: number
  disclaimer: boolean
}

export interface KnowledgeBaseStatus {
  total_documents: number
  total_chunks: number
  vector_store_exists: boolean
  embedding_model: string
  llm_model: string
}

export interface DocumentInfo {
  filename: string
  size: number
  type: string
}

export async function sendChat(request: ChatRequest): Promise<ChatResponse> {
  const res = await apiFetch('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })
  if (!res.ok) throw new Error(await readError(res, '请求失败'))
  return res.json()
}

export async function getKnowledgeBaseStatus(): Promise<KnowledgeBaseStatus> {
  const res = await apiFetch('/knowledge-base/status')
  if (!res.ok) throw new Error(await readError(res, '获取知识库状态失败'))
  return res.json()
}

export async function buildKnowledgeBase(rebuild = false): Promise<{ message: string }> {
  const res = await apiFetch('/knowledge-base/build', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ rebuild }),
  })
  if (!res.ok) throw new Error(await readError(res, '构建失败'))
  return res.json()
}

export async function uploadDocument(file: File): Promise<{ success: boolean; filename: string; chunk_count: number; message: string }> {
  const formData = new FormData()
  formData.append('file', file)
  // 不设 Content-Type：交给浏览器带 boundary
  const res = await apiFetch('/documents/upload', {
    method: 'POST',
    body: formData,
  })
  if (!res.ok) throw new Error(await readError(res, '上传失败'))
  return res.json()
}

export async function listDocuments(): Promise<{ documents: DocumentInfo[]; total: number }> {
  const res = await apiFetch('/documents/list')
  if (!res.ok) throw new Error(await readError(res, '获取文档列表失败'))
  return res.json()
}

export async function deleteDocument(filename: string): Promise<{ message: string }> {
  const res = await apiFetch(`/documents/${encodeURIComponent(filename)}`, {
    method: 'DELETE',
  })
  if (!res.ok) throw new Error(await readError(res, '删除失败'))
  return res.json()
}

// V2 评估接口
export interface EvalDetail {
  question: string
  question_type: string
  law: string
  answer: string
  ground_truth: string
  relevant_articles?: string[]
  source_count: number
  rag_mode: string
  retrieval?: Record<string, number>
  response?: Record<string, number>
  retrieved_ids?: string[]
  relevant_ids?: string[]
}

export interface TriadMetrics {
  faithfulness: number
  context_precision: number
  context_recall: number
}

export interface RetrievalMetrics {
  'precision@5': number
  'recall@5': number
  'f1@5': number
  mrr: number
  map: number
}

export interface ResponseMetrics {
  rouge_l: number
  bleu: number
  hallucination_rate?: number
  completeness?: number
}

export interface EvalResult {
  message: string
  rag_mode: string
  triad: TriadMetrics
  retrieval: RetrievalMetrics
  response: ResponseMetrics
  scores: Record<string, number>
  type_scores: Record<string, Record<string, number>>
  type_retrieval: Record<string, RetrievalMetrics>
  type_response: Record<string, ResponseMetrics>
  sample_count: number
  details: EvalDetail[]
  report_path: string
}

export interface EvalStatus {
  running: boolean
  progress: string
  result: EvalResult | null
  error: string | null
}

export interface EvalComparison {
  metrics: string[]
  results: Record<string, Record<string, number>>
  triad: Record<string, TriadMetrics>
  retrieval: Record<string, RetrievalMetrics>
  response: Record<string, ResponseMetrics>
  message?: string
}

export interface Observability {
  retrieval_quality: {
    'precision@1': number
    'recall@5': number
  }
  generation_quality: {
    faithfulness: number
    hallucination_rate: number
    completeness: number
  }
  business: {
    resolution_rate: number
    first_answer_usability: number
  }
  message?: string
}

export interface BadCase {
  question: string
  question_type: string
  law: string
  answer: string
  ground_truth: string
  source_count: number
  rag_mode: string
  retrieval?: Record<string, number>
  response?: Record<string, number>
}

export async function getEvaluationReport(): Promise<EvalComparison> {
  const res = await apiFetch('/evaluation/report')
  if (!res.ok) throw new Error(await readError(res, '获取评估报告失败'))
  return res.json()
}

export async function getObservability(): Promise<Observability> {
  const res = await apiFetch('/evaluation/observability')
  if (!res.ok) throw new Error(await readError(res, '获取可观测性数据失败'))
  return res.json()
}

export async function getBadCases(topN = 5): Promise<BadCase[]> {
  const res = await apiFetch(`/evaluation/bad-cases?top_n=${topN}`)
  if (!res.ok) throw new Error(await readError(res, '获取Bad Case失败'))
  return res.json()
}

// V3 断点续评接口
export interface PersistentTask {
  task_id: string
  status: 'created' | 'running' | 'scoring' | 'paused' | 'completed' | 'failed'
  rag_mode: string | null
  sample_count: number | null
  question_type: string | null
  total_questions: number
  completed_indices: number[]
  failed_indices: number[]
  created_at: string
  updated_at: string
  started_at: string | null
  finished_at: string | null
  error: string | null
}

export interface PersistentProgress {
  task_id: string
  status: string
  total: number
  completed: number
  failed: number
  remaining: number
  progress_pct: number
  rag_mode: string | null
  question_type: string | null
}

export interface PersistentReport {
  task_id: string
  status: string
  rag_mode: string | null
  question_type: string | null
  summary: {
    total: number
    success: number
    failed: number
    remaining: number
    success_rate: number
  }
  retrieval_avg?: Record<string, number>
  response_avg?: Record<string, number>
  failed_items?: Array<{ index: number; question: string; error: string }>
  ragas_full?: EvalResult
  created_at?: string
  started_at?: string
  finished_at?: string
}

export async function createPersistentTask(
  ragMode?: string,
  sampleCount?: number,
  questionType?: string,
  sampleOffset?: number,
  evalSubset?: string,
  difficulty?: string,
): Promise<{ task_id: string; message: string }> {
  const params = new URLSearchParams()
  if (ragMode) params.set('rag_mode', ragMode)
  if (sampleCount) params.set('sample_count', String(sampleCount))
  if (questionType) params.set('question_type', questionType)
  if (sampleOffset) params.set('sample_offset', String(sampleOffset))
  if (evalSubset) params.set('eval_subset', evalSubset)
  if (difficulty) params.set('difficulty', difficulty)
  const qs = params.toString() ? `?${params.toString()}` : ''
  const res = await apiFetch(`/evaluation/persistent/create${qs}`, {
    method: 'POST',
  })
  if (!res.ok) throw new Error(await readError(res, '创建任务失败'))
  return res.json()
}

async function _persistentPost(action: string, taskId: string, errorMsg: string): Promise<{ task_id: string; message: string; status: string }> {
  const res = await apiFetch(`/evaluation/persistent/${action}?task_id=${encodeURIComponent(taskId)}`, {
    method: 'POST',
  })
  if (!res.ok) throw new Error(await readError(res, errorMsg))
  return res.json()
}

export const startPersistentTask = (taskId: string) => _persistentPost('start', taskId, '启动任务失败')
export const resumePersistentTask = (taskId: string) => _persistentPost('resume', taskId, '续评失败')
export const restartPersistentTask = (taskId: string) => _persistentPost('restart', taskId, '重新评估失败')
export const retryFailedPersistentTask = (taskId: string) => _persistentPost('retry-failed', taskId, '重试失败')

export async function listPersistentTasks(): Promise<{ tasks: PersistentTask[] }> {
  const res = await apiFetch('/evaluation/persistent/tasks')
  if (!res.ok) throw new Error(await readError(res, '获取任务列表失败'))
  return res.json()
}

export async function getPersistentProgress(taskId: string): Promise<PersistentProgress> {
  const res = await apiFetch(`/evaluation/persistent/progress?task_id=${encodeURIComponent(taskId)}`)
  if (!res.ok) throw new Error(await readError(res, '获取进度失败'))
  return res.json()
}

export async function getPersistentReport(taskId: string): Promise<PersistentReport> {
  const res = await apiFetch(`/evaluation/persistent/report?task_id=${encodeURIComponent(taskId)}`)
  if (!res.ok) throw new Error(await readError(res, '获取报告失败'))
  return res.json()
}

export async function forceStopPersistentTask(taskId: string): Promise<{ task_id: string; message: string; status: string }> {
  const res = await apiFetch(`/evaluation/persistent/force-stop?task_id=${encodeURIComponent(taskId)}`, { method: 'POST' })
  if (!res.ok) throw new Error(await readError(res, '停止任务失败'))
  return res.json()
}

export async function deletePersistentTask(taskId: string): Promise<{ task_id: string; message: string }> {
  const res = await apiFetch(`/evaluation/persistent/delete?task_id=${encodeURIComponent(taskId)}`, { method: 'DELETE' })
  if (!res.ok) throw new Error(await readError(res, '删除任务失败'))
  return res.json()
}
