import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import { Plus, ChevronDown, ChevronRight, Zap } from 'lucide-react'
import MessageBubble from './MessageBubble'
import InputBar from './InputBar'
import { sendChat, type SourceDocument } from '../api'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: SourceDocument[]
  cragSteps?: string[]
  rewrittenQuestion?: string
  ragMode?: string
}

export default function ChatWindow() {
  const [messages, setMessages] = useState<Message[]>([])
  const [loading, setLoading] = useState(false)
  const [conversationId, setConversationId] = useState<string | undefined>(undefined)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleNewChat = () => {
    setMessages([])
    setConversationId(undefined)
  }

  const handleSend = async (text: string) => {
    const userMsg: Message = { id: Date.now().toString(), role: 'user', content: text }
    setMessages((prev) => [...prev, userMsg])
    setLoading(true)

    try {
      const res = await sendChat({ question: text, conversation_id: conversationId })
      // 保存conversation_id，后续请求复用
      if (!conversationId) {
        setConversationId(res.conversation_id)
      }
      const aiMsg: Message = {
        id: res.conversation_id,
        role: 'assistant',
        content: res.answer,
        sources: res.sources,
        cragSteps: res.crag_steps,
        rewrittenQuestion: res.rewritten_question,
        ragMode: res.rag_mode,
      }
      setMessages((prev) => [...prev, aiMsg])
    } catch (err) {
      const errMsg: Message = {
        id: 'err-' + Date.now(),
        role: 'assistant',
        content: `抱歉，请求出错：${(err as Error).message}`,
      }
      setMessages((prev) => [...prev, errMsg])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header with new chat button */}
      {messages.length > 0 && (
        <div className="flex items-center justify-between px-6 py-2 bg-white border-b border-gray-100">
          <span className="text-xs text-gray-400">
            对话ID: {conversationId?.slice(0, 8)}...
          </span>
          <button
            onClick={handleNewChat}
            className="flex items-center gap-1 text-xs text-gray-500 hover:text-emerald-600 transition"
          >
            <Plus size={14} />
            新建对话
          </button>
        </div>
      )}

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-gray-400 space-y-3">
            <div className="w-16 h-16 rounded-full bg-emerald-50 flex items-center justify-center">
              <span className="text-2xl">⚖️</span>
            </div>
            <p className="text-lg font-medium">劳动法智能问答系统</p>
            <p className="text-sm">输入劳动法相关问题，我将基于法律条文为您解答</p>
            <p className="text-xs text-blue-500 mt-1">V3 支持多轮对话，可连续追问</p>
            <p className="text-xs text-amber-500 mt-2">⚠️ 本系统回答仅供参考，不构成法律意见，具体问题请咨询专业律师</p>
          </div>
        )}

        {messages.map((msg) => (
          <div key={msg.id}>
            <MessageBubble role={msg.role} sources={msg.sources}>
              {msg.role === 'assistant' ? (
                <ReactMarkdown>{msg.content}</ReactMarkdown>
              ) : (
                msg.content
              )}
            </MessageBubble>
            {/* CRAG步骤可视化 */}
            {msg.role === 'assistant' && msg.cragSteps && msg.cragSteps.length > 0 && (
              <CragStepsPanel steps={msg.cragSteps} rewrittenQuestion={msg.rewrittenQuestion} />
            )}
          </div>
        ))}

        {loading && (
          <MessageBubble role="assistant">
            <div className="flex items-center gap-2 text-gray-400">
              <div className="flex gap-1">
                <span className="w-2 h-2 bg-gray-300 rounded-full animate-bounce [animation-delay:-0.3s]" />
                <span className="w-2 h-2 bg-gray-300 rounded-full animate-bounce [animation-delay:-0.15s]" />
                <span className="w-2 h-2 bg-gray-300 rounded-full animate-bounce" />
              </div>
              <span className="text-xs">正在检索法律条文...</span>
            </div>
          </MessageBubble>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Disclaimer */}
      <div className="px-6 py-1.5 bg-amber-50 border-t border-amber-100">
        <p className="text-xs text-amber-600 text-center">⚠️ 本系统回答仅供参考，不构成法律意见，具体问题请咨询专业律师</p>
      </div>

      {/* Input Bar */}
      <InputBar onSend={handleSend} disabled={loading} />
    </div>
  )
}

function CragStepsPanel({ steps, rewrittenQuestion }: { steps: string[]; rewrittenQuestion?: string }) {
  const [expanded, setExpanded] = useState(false)

  const stepIcon = (step: string) => {
    if (step.includes('检索')) return '🔍'
    if (step.includes('评估检索')) return '📋'
    if (step.includes('改写')) return '✏️'
    if (step.includes('生成')) return '💬'
    if (step.includes('评估回答')) return '✅'
    return '→'
  }

  return (
    <div className="ml-12 mt-1">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1.5 text-xs text-blue-500 hover:text-blue-700 transition"
      >
        <Zap size={12} />
        <span className="font-medium">CRAG 工作流</span>
        <span className="text-gray-400">({steps.length}步)</span>
        {expanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
      </button>

      {expanded && (
        <div className="mt-2 bg-blue-50 rounded-lg p-3 border border-blue-100 space-y-1.5">
          {rewrittenQuestion && (
            <div className="text-xs text-blue-700 bg-blue-100 rounded px-2 py-1 mb-2">
              ✏️ 查询改写: {rewrittenQuestion}
            </div>
          )}
          {steps.map((step, i) => (
            <div key={i} className="flex items-start gap-2 text-xs text-gray-600">
              <span className="flex-shrink-0">{stepIcon(step)}</span>
              <span>{step}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
