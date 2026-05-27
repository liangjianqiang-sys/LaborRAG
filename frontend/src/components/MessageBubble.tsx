import { useState, type ReactNode } from 'react'
import { FileText, Scale, ChevronDown, ChevronRight, User, Bookmark } from 'lucide-react'
import type { SourceDocument } from '../api'

interface MessageBubbleProps {
  role: 'user' | 'assistant'
  children: ReactNode
  sources?: SourceDocument[]
}

export default function MessageBubble({ role, children, sources }: MessageBubbleProps) {
  const isUser = role === 'user'
  const [sourcesExpanded, setSourcesExpanded] = useState(false)

  return (
    <div className={`flex gap-3 ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
      {/* ── 头像 ── */}
      <div
        className={`flex-shrink-0 w-9 h-9 rounded-xl flex items-center justify-center text-white text-sm shadow-sm ${
          isUser
            ? 'bg-gradient-to-br from-slate-600 to-slate-700 shadow-slate-300/40'
            : 'bg-gradient-to-br from-brand-500 to-accent-500 shadow-brand-200/40'
        }`}
      >
        {isUser ? <User size={18} /> : <Scale size={18} strokeWidth={2.2} />}
      </div>

      {/* ── 内容 ── */}
      <div className={`max-w-[75%] min-w-0 ${isUser ? 'items-end' : 'items-start'}`}>
        <div
          className={`rounded-2xl px-4 py-3 text-sm leading-relaxed ${
            isUser
              ? 'bg-gradient-to-br from-brand-500 to-brand-600 text-white rounded-tr-md shadow-md shadow-brand-200/30'
              : 'bg-white text-slate-800 border border-slate-200/80 rounded-tl-md shadow-sm'
          }`}
        >
          {children}
        </div>

        {/* ── 参考来源 ── */}
        {sources && sources.length > 0 && (
          <div className="mt-2">
            <button
              onClick={() => setSourcesExpanded(!sourcesExpanded)}
              className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-brand-600 transition-colors duration-200"
            >
              <Bookmark size={11} strokeWidth={1.8} />
              <span className="font-medium">参考来源 ({sources.length})</span>
              {sourcesExpanded ? <ChevronDown size={11} /> : <ChevronRight size={11} />}
            </button>

            {sourcesExpanded && (
              <div className="mt-1.5 space-y-1.5 animate-scale-in">
                {sources.map((src, i) => (
                  <div
                    key={i}
                    className="flex items-start gap-2.5 rounded-xl bg-brand-50/30 border border-brand-100/60 px-3.5 py-2.5 text-xs text-slate-600 hover:bg-brand-50/60 transition-colors"
                  >
                    <div className="flex-shrink-0 w-7 h-7 rounded-lg bg-brand-100 flex items-center justify-center">
                      <FileText size={14} className="text-brand-600" />
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-1.5 flex-wrap">
                        <span className="font-semibold text-slate-700 text-xs">
                          {src.source}
                        </span>
                        {src.page != null && (
                          <span className="text-slate-400 text-[10px] bg-slate-100 px-1.5 py-0.5 rounded border border-slate-200/60">
                            第{src.page}页
                          </span>
                        )}
                        <span className="text-brand-500 text-[10px] font-mono bg-brand-50 px-1.5 py-0.5 rounded border border-brand-200/60">
                          相关度 {(src.score * 100).toFixed(1)}%
                        </span>
                      </div>
                      <p className="mt-1 text-slate-500 leading-relaxed line-clamp-2">
                        {src.content}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
