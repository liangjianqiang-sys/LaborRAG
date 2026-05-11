import { useState, useEffect, type FormEvent } from 'react'
import {
  Database,
  RefreshCw,
  Upload,
  Trash2,
  FileText,
  CheckCircle,
  XCircle,
  Loader2,
  BarChart3,
} from 'lucide-react'
import {
  getKnowledgeBaseStatus,
  buildKnowledgeBase,
  uploadDocument,
  listDocuments,
  deleteDocument,
  type KnowledgeBaseStatus,
  type DocumentInfo,
} from '../api'
import EvalPanel from './EvalPanel'

export default function Sidebar() {
  const [status, setStatus] = useState<KnowledgeBaseStatus | null>(null)
  const [documents, setDocuments] = useState<DocumentInfo[]>([])
  const [building, setBuilding] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [message, setMessage] = useState<{ text: string; ok: boolean } | null>(null)

  const refresh = async () => {
    try {
      const [s, d] = await Promise.all([getKnowledgeBaseStatus(), listDocuments()])
      setStatus(s)
      setDocuments(d.documents)
    } catch {
      showMessage('获取状态失败', false)
    }
  }

  useEffect(() => {
    refresh()
  }, [])

  const showMessage = (text: string, ok: boolean) => {
    setMessage({ text, ok })
    setTimeout(() => setMessage(null), 3000)
  }

  const handleBuild = async (e: FormEvent) => {
    e.preventDefault()
    setBuilding(true)
    try {
      await buildKnowledgeBase(true)
      showMessage('知识库构建成功', true)
      await refresh()
    } catch (err) {
      showMessage((err as Error).message, false)
    } finally {
      setBuilding(false)
    }
  }

  const handleUpload = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    const fileInput = (e.target as HTMLFormElement).file as unknown as FileList
    const file = fileInput?.[0]
    if (!file) return

    setUploading(true)
    try {
      await uploadDocument(file)
      showMessage('文档上传成功', true)
      await refresh()
    } catch (err) {
      showMessage((err as Error).message, false)
    } finally {
      setUploading(false)
    }
  }

  const handleDelete = async (filename: string) => {
    if (!confirm(`确定删除 ${filename} 吗？`)) return
    try {
      await deleteDocument(filename)
      showMessage('删除成功', true)
      await refresh()
    } catch (err) {
      showMessage((err as Error).message, false)
    }
  }

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return bytes + ' B'
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
  }

  return (
    <aside className="w-72 flex-shrink-0 bg-gray-50 border-r border-gray-200 flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center gap-2 text-emerald-700 font-bold text-lg">
          <Database size={20} />
          <span>知识库管理</span>
        </div>
      </div>

      {/* Status */}
      <div className="p-4 border-b border-gray-200 space-y-2 text-sm">
        <div className="flex items-center justify-between">
          <span className="text-gray-500">向量库状态</span>
          {status?.vector_store_exists ? (
            <span className="flex items-center gap-1 text-emerald-600">
              <CheckCircle size={14} /> 已就绪
            </span>
          ) : (
            <span className="flex items-center gap-1 text-gray-400">
              <XCircle size={14} /> 未构建
            </span>
          )}
        </div>
        <div className="flex items-center justify-between text-gray-500">
          <span>文档数</span>
          <span className="text-gray-800">{status?.total_documents ?? '-'}</span>
        </div>
        <div className="flex items-center justify-between text-gray-500">
          <span>向量片段</span>
          <span className="text-gray-800">{status?.total_chunks ?? '-'}</span>
        </div>
        <div className="flex items-center justify-between text-gray-500">
          <span>Embedding</span>
          <span className="text-gray-800 text-xs truncate ml-2" title={status?.embedding_model}>
            {status?.embedding_model?.split('/').pop() ?? '-'}
          </span>
        </div>
      </div>

      {/* Actions */}
      <div className="p-4 border-b border-gray-200 space-y-3">
        <button
          onClick={handleBuild}
          disabled={building}
          className="w-full flex items-center justify-center gap-2 rounded-lg bg-emerald-600 px-3 py-2 text-sm text-white hover:bg-emerald-700 disabled:opacity-50 transition-colors"
        >
          {building ? <Loader2 size={16} className="animate-spin" /> : <RefreshCw size={16} />}
          {building ? '构建中...' : '构建/重建知识库'}
        </button>

        <form onSubmit={handleUpload} className="relative">
          <input
            type="file"
            name="file"
            accept=".pdf,.docx,.txt,.md"
            onChange={(e) => {
              if (e.currentTarget.files?.[0]) {
                const form = e.currentTarget.closest('form')!
                const formData = new FormData(form)
                const file = formData.get('file') as File
                setUploading(true)
                uploadDocument(file)
                  .then(() => { showMessage('上传成功', true); refresh() })
                  .catch((err) => showMessage((err as Error).message, false))
                  .finally(() => setUploading(false))
              }
            }}
            disabled={uploading}
            className="hidden"
            id="file-upload"
          />
          <label
            htmlFor="file-upload"
            className={`w-full flex items-center justify-center gap-2 rounded-lg border border-gray-300 px-3 py-2 text-sm text-gray-600 hover:bg-gray-100 cursor-pointer transition-colors ${uploading ? 'opacity-50 pointer-events-none' : ''}`}
          >
            {uploading ? <Loader2 size={16} className="animate-spin" /> : <Upload size={16} />}
            {uploading ? '上传中...' : '上传文档'}
          </label>
        </form>

        <button
          onClick={refresh}
          className="w-full flex items-center justify-center gap-2 rounded-lg border border-gray-300 px-3 py-2 text-sm text-gray-600 hover:bg-gray-100 transition-colors"
        >
          <RefreshCw size={14} />
          刷新状态
        </button>
      </div>

      {/* V2 评估面板 */}
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center gap-2 text-blue-700 font-semibold text-sm mb-3">
          <BarChart3 size={16} />
          <span>RAGAS 评估</span>
        </div>
        <EvalPanel />
      </div>

      {/* Document List */}
      <div className="flex-1 overflow-y-auto p-4">
        <p className="text-xs text-gray-400 font-medium mb-2">文档列表</p>
        {documents.length === 0 ? (
          <p className="text-xs text-gray-400">暂无文档</p>
        ) : (
          <ul className="space-y-1">
            {documents.map((doc) => (
              <li
                key={doc.filename}
                className="flex items-center gap-2 rounded-lg px-2 py-1.5 text-xs hover:bg-gray-100 group"
              >
                <FileText size={14} className="flex-shrink-0 text-gray-400" />
                <span className="flex-1 truncate text-gray-700" title={doc.filename}>
                  {doc.filename}
                </span>
                <span className="text-gray-400">{formatSize(doc.size)}</span>
                <button
                  onClick={() => handleDelete(doc.filename)}
                  className="opacity-0 group-hover:opacity-100 text-red-400 hover:text-red-600 transition-opacity"
                  title="删除"
                >
                  <Trash2 size={14} />
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Message Toast */}
      {message && (
        <div
          className={`mx-4 mb-4 rounded-lg px-3 py-2 text-xs text-center ${
            message.ok ? 'bg-emerald-50 text-emerald-700' : 'bg-red-50 text-red-700'
          }`}
        >
          {message.text}
        </div>
      )}
    </aside>
  )
}
