const API_BASE = '/api/v1'

export interface ChatRequest {
  question: string
  conversation_id?: string
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
  rag_mode: string
  crag_steps: string[]
  rewritten_question: string
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
  const res = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || '请求失败')
  }
  return res.json()
}

export async function getKnowledgeBaseStatus(): Promise<KnowledgeBaseStatus> {
  const res = await fetch(`${API_BASE}/knowledge-base/status`)
  if (!res.ok) throw new Error('获取知识库状态失败')
  return res.json()
}

export async function buildKnowledgeBase(rebuild = false): Promise<{ message: string }> {
  const res = await fetch(`${API_BASE}/knowledge-base/build`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ rebuild }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || '构建失败')
  }
  return res.json()
}

export async function uploadDocument(file: File): Promise<{ success: boolean; filename: string; chunk_count: number; message: string }> {
  const formData = new FormData()
  formData.append('file', file)
  const res = await fetch(`${API_BASE}/documents/upload`, {
    method: 'POST',
    body: formData,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || '上传失败')
  }
  return res.json()
}

export async function listDocuments(): Promise<{ documents: DocumentInfo[]; total: number }> {
  const res = await fetch(`${API_BASE}/documents/list`)
  if (!res.ok) throw new Error('获取文档列表失败')
  return res.json()
}

export async function deleteDocument(filename: string): Promise<{ message: string }> {
  const res = await fetch(`${API_BASE}/documents/${encodeURIComponent(filename)}`, {
    method: 'DELETE',
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || '删除失败')
  }
  return res.json()
}

// V2 评估接口
export interface EvalResult {
  message: string
  retriever_type: string
  scores: Record<string, number>
  sample_count: number
  report_path: string
}

export interface EvalComparison {
  metrics: string[]
  results: Record<string, Record<string, number>>
  message?: string
}

export async function runEvaluation(sampleCount?: number): Promise<EvalResult> {
  const params = sampleCount ? `?sample_count=${sampleCount}` : ''
  const res = await fetch(`${API_BASE}/evaluation/run${params}`, {
    method: 'POST',
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || '评估失败')
  }
  return res.json()
}

export async function getEvaluationReport(): Promise<EvalComparison> {
  const res = await fetch(`${API_BASE}/evaluation/report`)
  if (!res.ok) throw new Error('获取评估报告失败')
  return res.json()
}
