import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import { Plus, ChevronDown, ChevronRight, Zap, Sparkles, Scale, Gavel, BookOpen, Brain, Calculator, AlertCircle, FileWarning } from 'lucide-react'
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

const SUGGESTED_QUESTIONS = [
  { icon: Gavel, label: '法条查询', color: 'brand', text: '劳动合同法第47条怎么规定的？' },
  { icon: Calculator, label: '金额计算', color: 'accent', text: '月薪8000加班10小时加班费多少？' },
  { icon: BookOpen, label: '对比分析', color: 'cyan', text: '经济补偿金和赔偿金有什么区别？' },
  { icon: AlertCircle, label: '实务问题', color: 'amber', text: '用人单位可以随时辞退员工吗？' },
  { icon: Brain, label: '金额计算', color: 'accent', text: '工作5年被违法辞退赔偿金多少？' },
  { icon: FileWarning, label: '实务问题', color: 'amber', text: '未签劳动合同有什么法律后果？' },
]

const TAG_STYLES: Record<string, string> = {
  brand: 'bg-brand-50 text-brand-600 border-brand-200',
  accent: 'bg-accent-50 text-accent-600 border-accent-200',
  cyan: 'bg-cyan-50 text-cyan-600 border-cyan-100',
  amber: 'bg-amber-50 text-amber-600 border-amber-200',
}

const TAG_ICON_BG: Record<string, string> = {
  brand: 'bg-brand-100 group-hover:bg-brand-200',
  accent: 'bg-accent-100 group-hover:bg-accent-200',
  cyan: 'bg-cyan-100 group-hover:bg-cyan-100',
  amber: 'bg-amber-100 group-hover:bg-amber-200',
}

const TAG_ICON_COLOR: Record<string, string> = {
  brand: 'text-brand-500 group-hover:text-brand-700',
  accent: 'text-accent-500 group-hover:text-accent-700',
  cyan: 'text-cyan-500 group-hover:text-cyan-700',
  amber: 'text-amber-500 group-hover:text-amber-700',
}

const RAG_MODE_COLORS: Record<string, string> = {
  simple: 'bg-brand-50 text-brand-700 border-brand-200',
  crag: 'bg-accent-50 text-accent-600 border-accent-200',
  agent: 'bg-cyan-50 text-cyan-600 border-cyan-100',
}

const RAG_MODE_LABELS: Record<string, string> = {
  simple: 'V1/V2',
  crag: 'V3 CRAG',
  agent: 'V4 Agent',
}

export default function ChatWindow() {
  const [messages, setMessages] = useState<Message[]>([])
  const [loading, setLoading] = useState(false)
  const [conversationId, setConversationId] = useState<string | undefined>(undefined)
  const [isThinking, setIsThinking] = useState(false)
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
    setIsThinking(true)

    // Simulate thinking phases
    const thinkingTimer = setTimeout(() => setIsThinking(false), 800)

    try {
      const res = await sendChat({ question: text, conversation_id: conversationId })
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
      clearTimeout(thinkingTimer)
    }
  }

  return (
    <div className="flex flex-col h-full">
      {/* ── 对话工具栏 ── */}
      {messages.length > 0 && (
        <div className="flex items-center justify-between px-5 lg:px-6 py-2.5 bg-white/70 border-b border-brand-100/40 backdrop-blur-sm">
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <span className="w-1.5 h-1.5 rounded-full bg-brand-400 animate-glow-pulse" />
            对话ID: <span className="font-mono">{conversationId?.slice(0, 8)}...</span>
          </div>
          <button
            onClick={handleNewChat}
            className="flex items-center gap-1.5 text-xs font-medium text-slate-500 hover:text-brand-600 px-2.5 py-1.5 rounded-md hover:bg-brand-50 transition-all duration-200"
          >
            <Plus size={14} strokeWidth={2} />
            新建对话
          </button>
        </div>
      )}

      {/* ── 消息区域 ── */}
      <div className="flex-1 overflow-y-auto px-4 lg:px-6 py-5 space-y-5 scroll-smooth">
        {messages.length === 0 ? (
          <WelcomeScreen onSend={handleSend} />
        ) : (
          <>
            {messages.map((msg) => (
              <div key={msg.id} className="animate-slide-up">
                <MessageBubble role={msg.role} sources={msg.sources}>
                  {msg.role === 'assistant' ? (
                    <div className="prose-content">
                      <ReactMarkdown>{msg.content}</ReactMarkdown>
                    </div>
                  ) : (
                    <p className="leading-relaxed">{msg.content}</p>
                  )}
                </MessageBubble>

                {/* RAG 模式标签 + 步骤面板 */}
                {msg.role === 'assistant' && (
                  <div className="ml-12 lg:ml-14 mt-1.5 space-y-1">
                    {msg.ragMode && RAG_MODE_LABELS[msg.ragMode] && (
                      <span
                        className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-semibold border ${
                          RAG_MODE_COLORS[msg.ragMode] || 'bg-gray-100 text-gray-600 border-gray-200'
                        }`}
                      >
                        <Zap size={10} />
                        {RAG_MODE_LABELS[msg.ragMode]}
                      </span>
                    )}
                    {msg.cragSteps && msg.cragSteps.length > 0 && (
                      <CragStepsPanel steps={msg.cragSteps} rewrittenQuestion={msg.rewrittenQuestion} />
                    )}
                    {/* 免责声明 */}
                    <p className="text-[11px] text-amber-500/80 ml-0.5">
                      ⚠️ 仅供参考，不构成法律意见
                    </p>
                  </div>
                )}
              </div>
            ))}

            {/* 加载状态 */}
            {loading && (
              <div className="animate-slide-up">
                <MessageBubble role="assistant">
                  <div className="flex items-center gap-2.5 py-0.5">
                    {isThinking ? (
                      <>
                        <div className="thinking-indicator">正在思考</div>
                      </>
                    ) : (
                      <>
                        <div className="flex gap-1">
                          <span className="typing-dot" />
                          <span className="typing-dot" />
                          <span className="typing-dot" />
                        </div>
                        <span className="text-xs text-slate-400 font-medium">
                          正在检索法律条文...
                        </span>
                      </>
                    )}
                  </div>
                </MessageBubble>
              </div>
            )}
          </>
        )}

        <div ref={bottomRef} />
      </div>

      {/* ── 输入栏 ── */}
      <InputBar onSend={handleSend} disabled={loading} />
    </div>
  )
}

/* ── 欢迎页面 ── */
function WelcomeScreen({ onSend }: { onSend: (text: string) => void }) {
  return (
    <div className="flex flex-col items-center justify-center h-full text-slate-400 space-y-6 animate-fade-in">
      {/* Logo 区域 — 蓝渐变光环 */}
      <div className="relative animate-scale-in">
        <div className="absolute inset-0 w-20 h-20 rounded-2xl bg-gradient-to-br from-brand-400/30 to-accent-400/30 blur-xl scale-150" />
        <div className="relative w-20 h-20 rounded-2xl bg-gradient-to-br from-brand-500 to-accent-500 flex items-center justify-center shadow-lg shadow-brand-300/30">
          <Scale size={36} className="text-white" strokeWidth={2} />
        </div>
        <div className="absolute -top-1 -right-1 w-7 h-7 rounded-full bg-white shadow-md flex items-center justify-center">
          <Sparkles size={14} className="text-brand-500" />
        </div>
      </div>

      <div className="text-center animate-slide-up">
        <h1 className="text-xl font-bold text-gradient-brand tracking-tight">劳动法智能问答系统</h1>
        <p className="text-sm text-slate-400 mt-1.5 max-w-sm leading-relaxed">
          基于 RAG 技术的劳动法知识助手，输入问题即可获得基于法律条文的专业解答
        </p>
      </div>

      {/* 快速问题 — 图标卡片式布局 */}
      <div className="w-full max-w-lg mt-1 animate-slide-up" style={{ animationDelay: '0.1s' }}>
        <p className="text-xs text-slate-400 text-center mb-3 flex items-center justify-center gap-1.5">
          <span className="h-px w-6 bg-brand-200" />
          <Sparkles size={12} className="text-brand-400" />
          试试这些问题
          <Sparkles size={12} className="text-brand-400" />
          <span className="h-px w-6 bg-brand-200" />
        </p>
        <div className="grid grid-cols-2 gap-2.5">
          {SUGGESTED_QUESTIONS.map((q, i) => (
            <button
              key={i}
              onClick={() => onSend(q.text)}
              className="group relative flex items-start gap-3 p-3.5 rounded-xl border border-slate-200/80 bg-white hover:bg-brand-50/40 hover:border-brand-300/60 transition-all duration-200 text-left shadow-sm hover:shadow-card-hover animate-scale-in"
              style={{ animationDelay: `${0.05 * i}s` }}
            >
              <div className={`flex-shrink-0 w-8 h-8 rounded-lg ${TAG_ICON_BG[q.color]} flex items-center justify-center transition-colors duration-200`}>
                <q.icon size={16} className={`${TAG_ICON_COLOR[q.color]} transition-colors duration-200`} />
              </div>
              <div className="min-w-0">
                <span className={`inline-block text-[10px] font-semibold px-1.5 py-0.5 rounded border mb-1 ${TAG_STYLES[q.color]}`}>
                  {q.label}
                </span>
                <p className="text-xs text-slate-600 group-hover:text-slate-800 leading-relaxed transition-colors duration-200">
                  {q.text}
                </p>
              </div>
            </button>
          ))}
        </div>
      </div>

      <div className="flex items-center gap-4 text-[11px] text-slate-400 pt-1 animate-fade-in" style={{ animationDelay: '0.3s' }}>
        <span className="flex items-center gap-1">
          <span className="w-1.5 h-1.5 rounded-full bg-brand-400" />
          支持多轮对话
        </span>
        <span className="flex items-center gap-1">
          <span className="w-1.5 h-1.5 rounded-full bg-accent-400" />
          引用法律原文
        </span>
        <span className="flex items-center gap-1">
          <span className="w-1.5 h-1.5 rounded-full bg-cyan-500" />
          CRAG 纠错
        </span>
      </div>
    </div>
  )
}

/* ── CRAG 步骤面板 ── */
function CragStepsPanel({ steps, rewrittenQuestion }: { steps: string[]; rewrittenQuestion?: string }) {
  const [expanded, setExpanded] = useState(false)

  const stepMeta = (step: string) => {
    if (step.includes('意图路由'))    return { icon: '🧭', label: '意图路由' }
    if (step.includes('检索'))        return { icon: '🔍', label: '检索' }
    if (step.includes('评估检索'))    return { icon: '📋', label: '检索评估' }
    if (step.includes('改写'))        return { icon: '✏️', label: '查询改写' }
    if (step.includes('计算'))        return { icon: '🧮', label: '计算' }
    if (step.includes('对比'))        return { icon: '⚖️', label: '对比' }
    if (step.includes('验证'))        return { icon: '✅', label: '验证' }
    if (step.includes('生成'))        return { icon: '💬', label: '生成' }
    if (step.includes('评估回答'))    return { icon: '✅', label: '回答评估' }
    return { icon: '→', label: '步骤' }
  }

  return (
    <div>
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-brand-600 transition-colors duration-200"
      >
        <Zap size={12} strokeWidth={1.8} />
        <span className="font-medium">RAG 工作流</span>
        <span className="text-slate-300">({steps.length} 步)</span>
        {expanded ? <ChevronDown size={11} /> : <ChevronRight size={11} />}
      </button>

      {expanded && (
        <div className="mt-2 bg-gradient-to-br from-brand-50/80 to-accent-50/80 rounded-xl p-3.5 border border-brand-200/60 space-y-1.5 shadow-sm animate-scale-in">
          {rewrittenQuestion && (
            <div className="flex items-start gap-2 text-xs text-brand-700 bg-brand-100/60 rounded-lg px-3 py-2 mb-2 border border-brand-200/50">
              <span className="flex-shrink-0">✏️</span>
              <div>
                <span className="font-semibold">查询改写: </span>
                <span>{rewrittenQuestion}</span>
              </div>
            </div>
          )}
          {steps.map((step, i) => {
            const meta = stepMeta(step)
            return (
              <div key={i} className="flex items-center gap-2.5 text-xs text-slate-600">
                <div className="flex items-center gap-1.5 flex-shrink-0">
                  <span className="flex-shrink-0 w-5 h-5 rounded-full bg-gradient-to-br from-brand-400 to-accent-400 flex items-center justify-center text-[10px] font-bold text-white shadow-sm">
                    {i + 1}
                  </span>
                  <span className="text-[11px]">{meta.icon}</span>
                </div>
                <span className="text-slate-500">{step}</span>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
