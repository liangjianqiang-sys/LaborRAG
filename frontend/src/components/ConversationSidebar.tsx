import { useState } from 'react'
import { Plus, MessageSquare, Trash2, PanelLeftClose, PanelLeft } from 'lucide-react'
import type { Conversation } from '../hooks/useConversations'

interface ConversationSidebarProps {
  conversations: Conversation[]
  activeId: string | null
  onSelect: (id: string) => void
  onNew: () => void
  onDelete: (id: string) => void
  collapsed: boolean
  onToggleCollapse: () => void
}

/** 格式化时间：今天/昨天/日期 */
function formatTime(ts: number): string {
  const d = new Date(ts)
  const now = new Date()
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime()
  const yesterday = today - 86400000
  if (ts >= today) return '今天'
  if (ts >= yesterday) return '昨天'
  return `${d.getMonth() + 1}/${d.getDate()}`
}

export default function ConversationSidebar({
  conversations,
  activeId,
  onSelect,
  onNew,
  onDelete,
  collapsed,
  onToggleCollapse,
}: ConversationSidebarProps) {
  const [hoveredId, setHoveredId] = useState<string | null>(null)

  // 按日期分组
  const groups: { label: string; items: Conversation[] }[] = []
  let currentLabel = ''
  let currentItems: Conversation[] = []

  for (const conv of conversations) {
    const label = formatTime(conv.updatedAt)
    if (label !== currentLabel) {
      if (currentItems.length > 0) groups.push({ label: currentLabel, items: currentItems })
      currentLabel = label
      currentItems = [conv]
    } else {
      currentItems.push(conv)
    }
  }
  if (currentItems.length > 0) groups.push({ label: currentLabel, items: currentItems })

  if (collapsed) {
    return (
      <div className="flex flex-col items-center py-3 px-1.5 w-12 bg-slate-50/80 border-r border-brand-100/40 flex-shrink-0">
        <button
          onClick={onToggleCollapse}
          className="w-8 h-8 rounded-lg flex items-center justify-center text-slate-400 hover:text-brand-600 hover:bg-brand-50 transition-all duration-200"
          title="展开侧边栏"
        >
          <PanelLeft size={16} />
        </button>
        <button
          onClick={onNew}
          className="mt-2 w-8 h-8 rounded-lg bg-gradient-to-br from-brand-500 to-accent-500 text-white flex items-center justify-center shadow-sm hover:shadow-md transition-all duration-200"
          title="新建对话"
        >
          <Plus size={16} />
        </button>
        <div className="mt-3 flex flex-col gap-1.5 overflow-y-auto flex-1">
          {conversations.map((conv) => (
            <button
              key={conv.id}
              onClick={() => onSelect(conv.id)}
              className={`w-8 h-8 rounded-lg flex items-center justify-center transition-all duration-200 ${
                conv.id === activeId
                  ? 'bg-brand-100 text-brand-600'
                  : 'text-slate-400 hover:text-brand-600 hover:bg-brand-50'
              }`}
              title={conv.title}
            >
              <MessageSquare size={14} />
            </button>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="flex flex-col w-64 bg-slate-50/80 border-r border-brand-100/40 flex-shrink-0 animate-slide-in-left">
      {/* ── 头部 ── */}
      <div className="flex items-center justify-between px-3 py-3 border-b border-brand-100/30">
        <span className="text-xs font-semibold text-slate-500 tracking-wide">历史对话</span>
        <div className="flex items-center gap-1">
          <button
            onClick={onNew}
            className="flex items-center gap-1 text-xs font-medium text-brand-600 hover:text-brand-700 px-2 py-1 rounded-md hover:bg-brand-50 transition-all duration-200"
          >
            <Plus size={13} strokeWidth={2.2} />
            新建
          </button>
          <button
            onClick={onToggleCollapse}
            className="w-6 h-6 rounded flex items-center justify-center text-slate-400 hover:text-brand-600 hover:bg-brand-50 transition-all duration-200"
            title="收起侧边栏"
          >
            <PanelLeftClose size={14} />
          </button>
        </div>
      </div>

      {/* ── 对话列表 ── */}
      <div className="flex-1 overflow-y-auto px-2 py-2 space-y-3">
        {groups.map((group) => (
          <div key={group.label}>
            <div className="px-2 py-1 text-[10px] font-semibold text-slate-400 uppercase tracking-wider">
              {group.label}
            </div>
            <div className="space-y-0.5">
              {group.items.map((conv) => (
                <div
                  key={conv.id}
                  onMouseEnter={() => setHoveredId(conv.id)}
                  onMouseLeave={() => setHoveredId(null)}
                  onClick={() => onSelect(conv.id)}
                  className={`group relative flex items-center gap-2 px-2.5 py-2 rounded-lg cursor-pointer transition-all duration-200 ${
                    conv.id === activeId
                      ? 'bg-brand-50 border border-brand-200/60 shadow-sm'
                      : 'hover:bg-white border border-transparent'
                  }`}
                >
                  <MessageSquare
                    size={13}
                    className={`flex-shrink-0 ${
                      conv.id === activeId ? 'text-brand-500' : 'text-slate-400'
                    }`}
                  />
                  <div className="min-w-0 flex-1">
                    <p
                      className={`text-xs truncate leading-snug ${
                        conv.id === activeId ? 'text-brand-700 font-medium' : 'text-slate-600'
                      }`}
                    >
                      {conv.title}
                    </p>
                    <p className="text-[10px] text-slate-400 mt-0.5">
                      {conv.messages.length} 条消息
                    </p>
                  </div>
                  {/* 删除按钮 */}
                  {hoveredId === conv.id && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        if (confirm('确定删除此对话？')) onDelete(conv.id)
                      }}
                      className="flex-shrink-0 w-5 h-5 rounded flex items-center justify-center text-slate-400 hover:text-red-500 hover:bg-red-50 transition-all duration-150 animate-fade-in"
                    >
                      <Trash2 size={11} />
                    </button>
                  )}
                </div>
              ))}
            </div>
          </div>
        ))}

        {conversations.length === 0 && (
          <div className="flex flex-col items-center justify-center py-8 text-slate-400">
            <MessageSquare size={24} className="mb-2 opacity-40" />
            <p className="text-xs">暂无对话</p>
            <p className="text-[10px] mt-1">点击"新建"开始</p>
          </div>
        )}
      </div>
    </div>
  )
}
