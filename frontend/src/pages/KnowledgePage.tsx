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
  FolderOpen,
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

export default function KnowledgePage() {
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

  useEffect(() => { refresh() }, [])

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

  const handleUpload = async (file: File) => {
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
    <div className="max-w-4xl mx-auto p-6 space-y-6">
      <h2 className="text-2xl font-bold text-gray-800 flex items-center gap-2">
        <Database size={24} className="text-emerald-600" />
        知识库管理
      </h2>

      {/* 状态卡片 */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-white rounded-xl p-4 border border-gray-200">
          <p className="text-xs text-gray-500 mb-1">向量库状态</p>
          {status?.vector_store_exists ? (
            <span className="flex items-center gap-1 text-emerald-600 font-semibold">
              <CheckCircle size={16} /> 已就绪
            </span>
          ) : (
            <span className="flex items-center gap-1 text-gray-400 font-semibold">
              <XCircle size={16} /> 未构建
            </span>
          )}
        </div>
        <div className="bg-white rounded-xl p-4 border border-gray-200">
          <p className="text-xs text-gray-500 mb-1">文档数</p>
          <p className="text-xl font-bold text-gray-800">{status?.total_documents ?? '-'}</p>
        </div>
        <div className="bg-white rounded-xl p-4 border border-gray-200">
          <p className="text-xs text-gray-500 mb-1">向量片段</p>
          <p className="text-xl font-bold text-gray-800">{status?.total_chunks ?? '-'}</p>
        </div>
        <div className="bg-white rounded-xl p-4 border border-gray-200">
          <p className="text-xs text-gray-500 mb-1">Embedding</p>
          <p className="text-sm font-semibold text-gray-700 truncate" title={status?.embedding_model}>
            {status?.embedding_model?.split('/').pop() ?? '-'}
          </p>
        </div>
      </div>

      {/* 操作区 */}
      <div className="flex flex-wrap gap-3">
        <button
          onClick={handleBuild}
          disabled={building}
          className="flex items-center gap-2 rounded-lg bg-emerald-600 px-4 py-2.5 text-sm text-white hover:bg-emerald-700 disabled:opacity-50 transition-colors"
        >
          {building ? <Loader2 size={16} className="animate-spin" /> : <RefreshCw size={16} />}
          {building ? '构建中...' : '构建/重建知识库'}
        </button>

        <label className={`flex items-center gap-2 rounded-lg border border-gray-300 px-4 py-2.5 text-sm text-gray-600 hover:bg-gray-50 cursor-pointer transition-colors ${uploading ? 'opacity-50 pointer-events-none' : ''}`}>
          {uploading ? <Loader2 size={16} className="animate-spin" /> : <Upload size={16} />}
          {uploading ? '上传中...' : '上传文档'}
          <input
            type="file"
            accept=".pdf,.docx,.txt,.md"
            className="hidden"
            onChange={(e) => {
              const file = e.currentTarget.files?.[0]
              if (file) handleUpload(file)
            }}
          />
        </label>

        <button
          onClick={refresh}
          className="flex items-center gap-2 rounded-lg border border-gray-300 px-4 py-2.5 text-sm text-gray-600 hover:bg-gray-50 transition-colors"
        >
          <RefreshCw size={14} />
          刷新状态
        </button>
      </div>

      {/* 文档列表 */}
      <div className="bg-white rounded-xl border border-gray-200">
        <div className="px-4 py-3 border-b border-gray-100 flex items-center gap-2">
          <FolderOpen size={16} className="text-gray-400" />
          <span className="text-sm font-medium text-gray-700">文档列表</span>
          <span className="text-xs text-gray-400">({documents.length} 个文件)</span>
        </div>
        {documents.length === 0 ? (
          <div className="p-8 text-center text-gray-400 text-sm">暂无文档，请上传劳动法文档</div>
        ) : (
          <ul className="divide-y divide-gray-100">
            {documents.map((doc) => (
              <li key={doc.filename} className="flex items-center gap-3 px-4 py-3 hover:bg-gray-50 group">
                <FileText size={18} className="flex-shrink-0 text-gray-400" />
                <span className="flex-1 text-sm text-gray-700 truncate" title={doc.filename}>
                  {doc.filename}
                </span>
                <span className="text-xs text-gray-400">{formatSize(doc.size)}</span>
                <button
                  onClick={() => handleDelete(doc.filename)}
                  className="opacity-0 group-hover:opacity-100 text-red-400 hover:text-red-600 transition-opacity"
                  title="删除"
                >
                  <Trash2 size={16} />
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* 消息提示 */}
      {message && (
        <div className={`fixed bottom-6 left-1/2 -translate-x-1/2 rounded-lg px-4 py-2 text-sm shadow-lg ${
          message.ok ? 'bg-emerald-600 text-white' : 'bg-red-600 text-white'
        }`}>
          {message.text}
        </div>
      )}
    </div>
  )
}
