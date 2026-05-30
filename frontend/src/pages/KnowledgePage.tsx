import { useState, useEffect } from 'react'
import {
  Database,
  RefreshCw,

  Trash2,
  FileText,
  CheckCircle,
  XCircle,
  Loader2,
  FolderOpen,
  BookOpen,
  Layers,
  Cpu,
  FileUp,
  HardDrive,
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

  const handleBuild = async () => {
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

  const statCards = [
    {
      label: '向量库状态',
      value: status?.vector_store_exists ? '已就绪' : '未构建',
      icon: Database,
      color: status?.vector_store_exists
        ? 'text-emerald-600 bg-emerald-50'
        : 'text-slate-400 bg-slate-50',
      borderClass: status?.vector_store_exists ? 'border-emerald-100' : 'border-slate-200',
      iconColor: status?.vector_store_exists ? 'text-emerald-500' : 'text-slate-400',
      iconBg: status?.vector_store_exists ? 'bg-emerald-100' : 'bg-slate-100',
    },
    {
      label: '文档数量',
      value: status?.total_documents ?? '-',
      icon: BookOpen,
      color: 'text-brand-600 bg-brand-50',
      borderClass: 'border-brand-100',
      iconColor: 'text-brand-500',
      iconBg: 'bg-brand-100',
    },
    {
      label: '向量片段',
      value: status?.total_chunks ?? '-',
      icon: Layers,
      color: 'text-accent-600 bg-accent-50',
      borderClass: 'border-accent-200',
      iconColor: 'text-accent-500',
      iconBg: 'bg-accent-100',
    },
    {
      label: 'Embedding 模型',
      value: status?.embedding_model?.split('/').pop() ?? '-',
      icon: Cpu,
      color: 'text-cyan-600 bg-cyan-50',
      borderClass: 'border-cyan-100',
      iconColor: 'text-cyan-500',
      iconBg: 'bg-cyan-100',
      tooltip: status?.embedding_model,
    },
  ]

  return (
    <div className="max-w-4xl mx-auto p-5 lg:p-6 space-y-5 overflow-y-auto h-full">
      {/* ── 页面标题 ── */}
      <div className="flex items-center gap-3">
        <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-brand-500 to-accent-500 flex items-center justify-center shadow-sm shadow-brand-200/40">
          <Database size={20} className="text-white" />
        </div>
        <div>
          <h2 className="text-lg font-bold text-gradient-brand">知识库管理</h2>
          <p className="text-xs text-slate-400 mt-0.5">管理文档和向量库，构建检索基础</p>
        </div>
      </div>

      {/* ── 状态卡片网格 ── */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {statCards.map((card, i) => (
          <div
            key={i}
            className={`rounded-xl border p-4 card-base ${card.color} ${card.borderClass}`}
          >
            <div className="flex items-center gap-2.5">
              <div className={`w-9 h-9 rounded-lg ${card.iconBg} flex items-center justify-center`}>
                <card.icon size={18} className={card.iconColor} />
              </div>
              <div className="min-w-0">
                <p className="text-[11px] text-slate-500 font-medium">{card.label}</p>
                <p
                  className={`text-lg font-bold mt-0.5 ${card.color.split(' ')[0]}`}
                  {...(card.tooltip ? { title: card.tooltip } : {})}
                >
                  {typeof card.value === 'number' ? card.value.toLocaleString() : card.value}
                </p>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* ── 操作按钮 ── */}
      <div className="flex flex-wrap gap-2.5">
        <button
          onClick={handleBuild}
          disabled={building}
          className="inline-flex items-center gap-2 rounded-lg btn-brand px-4 py-2.5 text-sm disabled:opacity-50 transition-all duration-200"
        >
          {building ? <Loader2 size={16} className="animate-spin" /> : <RefreshCw size={16} />}
          {building ? '构建中...' : '构建 / 重建知识库'}
        </button>

        <label
          className={`inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-4 py-2.5 text-sm font-medium text-slate-600 hover:bg-brand-50 hover:border-brand-200 cursor-pointer transition-all duration-200 ${
            uploading ? 'opacity-50 pointer-events-none' : ''
          }`}
        >
          {uploading ? <Loader2 size={16} className="animate-spin" /> : <FileUp size={16} />}
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
          className="inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-4 py-2.5 text-sm font-medium text-slate-600 hover:bg-brand-50 hover:border-brand-200 transition-all duration-200"
        >
          <RefreshCw size={14} />
          刷新状态
        </button>
      </div>

      {/* ── 文档列表 ── */}
      <div className="bg-white rounded-xl border border-slate-200/80 shadow-card overflow-hidden">
        <div className="px-4 py-3.5 border-b border-brand-100/50 flex items-center gap-2.5 bg-brand-50/30">
          <FolderOpen size={16} className="text-brand-400" />
          <span className="text-sm font-semibold text-slate-700">文档列表</span>
          <span className="text-[11px] text-brand-500 bg-brand-50 px-2 py-0.5 rounded-full border border-brand-200/60">
            {documents.length} 个文件
          </span>
        </div>

        {documents.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-12 text-slate-400">
            <HardDrive size={32} className="text-brand-200 mb-3" />
            <p className="text-sm">暂无文档，请上传劳动法相关文档</p>
            <p className="text-xs text-slate-300 mt-1">支持 PDF、DOCX、TXT、MD 格式</p>
          </div>
        ) : (
          <ul className="divide-y divide-slate-100">
            {documents.map((doc) => {
              const ext = doc.filename.split('.').pop()?.toLowerCase() || ''
              const typeBadge: Record<string, string> = {
                pdf: 'bg-red-50 text-red-500 border-red-200',
                docx: 'bg-brand-50 text-brand-500 border-brand-200',
                txt: 'bg-slate-50 text-slate-500 border-slate-200',
                md: 'bg-accent-50 text-accent-500 border-accent-200',
              }
              const badgeClass = typeBadge[ext] || 'bg-slate-50 text-slate-500 border-slate-200'
              return (
              <li
                key={doc.filename}
                className="flex items-center gap-3 px-4 py-3.5 hover:bg-brand-50/30 group transition-colors"
              >
                <div className="w-8 h-8 rounded-lg bg-brand-50 flex items-center justify-center flex-shrink-0">
                  <FileText size={16} className="text-brand-500" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-slate-700 truncate font-medium" title={doc.filename}>
                    {doc.filename}
                  </p>
                  <div className="flex items-center gap-1.5 mt-0.5">
                    <span className={`text-[10px] font-semibold px-1.5 py-0.5 rounded border ${badgeClass}`}>
                      {ext.toUpperCase()}
                    </span>
                    <span className="text-[11px] text-slate-400">{formatSize(doc.size)}</span>
                  </div>
                </div>
                <button
                  onClick={() => handleDelete(doc.filename)}
                  className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs text-slate-400 opacity-0 group-hover:opacity-100 hover:bg-red-50 hover:text-red-500 transition-all duration-200"
                  title="删除"
                >
                  <Trash2 size={14} />
                  删除
                </button>
              </li>
            )})}
          </ul>
        )}
      </div>

      {/* ── Toast 消息 ── */}
      {message && (
        <div
          className={`fixed bottom-6 left-1/2 -translate-x-1/2 rounded-xl px-5 py-2.5 text-sm font-medium shadow-toast animate-fade-in z-50 ${
            message.ok
              ? 'bg-gradient-to-r from-brand-500 to-accent-500 text-white'
              : 'bg-gradient-to-r from-red-500 to-red-600 text-white'
          }`}
        >
          <div className="flex items-center gap-2">
            {message.ok ? <CheckCircle size={16} /> : <XCircle size={16} />}
            {message.text}
          </div>
        </div>
      )}
    </div>
  )
}
