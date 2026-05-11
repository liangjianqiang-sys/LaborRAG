import { useState, type ReactNode } from 'react'
import { FileText, Scale, ChevronDown, ChevronRight } from 'lucide-react'
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
      {/* Avatar */}
      <div
        className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-white text-sm ${
          isUser ? 'bg-blue-500' : 'bg-emerald-600'
        }`}
      >
        {isUser ? '你' : <Scale size={16} />}
      </div>

      {/* Content */}
      <div className={`max-w-[75%] ${isUser ? 'items-end' : 'items-start'}`}>
        <div
          className={`rounded-2xl px-4 py-3 text-sm leading-relaxed ${
            isUser
              ? 'bg-blue-500 text-white rounded-br-md'
              : 'bg-white text-gray-800 border border-gray-200 rounded-bl-md shadow-sm'
          }`}
        >
          {children}
        </div>

        {/* Sources - collapsible */}
        {sources && sources.length > 0 && (
          <div className="mt-2">
            <button
              onClick={() => setSourcesExpanded(!sourcesExpanded)}
              className="flex items-center gap-1 text-xs text-gray-400 hover:text-gray-600 transition"
            >
              <FileText size={12} />
              <span>参考来源 ({sources.length})</span>
              {sourcesExpanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
            </button>
            {sourcesExpanded && (
              <div className="mt-1 space-y-1">
                {sources.map((src, i) => (
                  <div
                    key={i}
                    className="flex items-start gap-2 rounded-lg bg-gray-50 border border-gray-100 px-3 py-2 text-xs text-gray-600"
                  >
                    <FileText size={14} className="flex-shrink-0 mt-0.5 text-gray-400" />
                    <div className="min-w-0">
                      <span className="font-medium text-gray-700">
                        {src.source}
                      </span>
                      {src.page != null && (
                        <span className="text-gray-400 ml-1">第{src.page}页</span>
                      )}
                      <span className="text-gray-400 ml-1">
                        (相关度: {(src.score * 100).toFixed(1)}%)
                      </span>
                      <p className="mt-0.5 text-gray-500 line-clamp-2">
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
