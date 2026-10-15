/**
 * 全局对话状态 —— 把「对话数据」与「发送请求」一起提到 App 层。
 *
 * 为什么必须放在这里，而不是 ChatPage 内部（2026-10-15 修复）
 * -------------------------------------------------------------
 * 原先 `useConversations()` 在 ChatPage 里调用，而 App.tsx 用的是
 * `isChat ? <ChatPage /> : <Routes>…` 条件渲染 —— 切到知识库 / 评估页时
 * **ChatPage 被整个卸载**，于是：
 *
 *   1. 对话状态随组件销毁（只剩 localStorage，而它只在消息落库那一刻写入）
 *   2. 正在跑的 /chat 请求，其 `onAddMessage` 回调指向已卸载组件的 setState，
 *      React 18 里是静默 no-op —— **回答被丢弃，也没写进 localStorage**
 *
 * 把状态和「发送」一起提到 Provider（挂在 App 上、永不卸载）后，
 * 切换页面不再影响生成过程，切回来还能看到正在流式输出的内容。
 */
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from 'react'
import { sendChatStream, type ChatHistoryItem } from '../api'

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: { content: string; source: string; score: number; page?: number }[]
  ragSteps?: string[]
  rewrittenQuestion?: string
  confidence?: number
  disclaimer?: boolean
  /** 该条回答仍在流式生成中（用于显示光标 / 禁用输入） */
  streaming?: boolean
  /** 本轮已走过的节点标签（流式期间实时展示进度） */
  liveSteps?: string[]
}

export interface Conversation {
  id: string
  title: string
  messages: Message[]
  conversationId?: string // 后端返回的 conversation_id
  createdAt: number
  updatedAt: number
}

const STORAGE_KEY = 'laborrag_conversations'

function loadConversations(): Conversation[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    const parsed = raw ? JSON.parse(raw) : []
    // 上次退出时若正好在生成中，恢复后要把 streaming 标记清掉，
    // 否则会出现一条永远转圈的空气泡
    return (parsed as Conversation[]).map((c) => ({
      ...c,
      messages: c.messages.map((m) => ({ ...m, streaming: false })),
    }))
  } catch {
    return []
  }
}

function saveConversations(conversations: Conversation[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations))
  } catch {
    // localStorage 满了（配额）不该让整个应用崩掉
  }
}

/** 从消息中自动提取对话标题（取第一条用户消息的前 20 字） */
function deriveTitle(messages: Message[]): string {
  const first = messages.find((m) => m.role === 'user')
  if (!first) return '新对话'
  const text = first.content.trim()
  return text.length > 20 ? text.slice(0, 20) + '…' : text
}

interface ConversationContextValue {
  conversations: Conversation[]
  activeId: string | null
  activeConversation: Conversation | null
  /** 是否有对话正在生成（跨页面有效） */
  isStreaming: boolean
  createConversation: () => string
  switchConversation: (id: string) => void
  deleteConversation: (id: string) => void
  clearActiveMessages: () => void
  sendMessage: (text: string) => Promise<void>
}

const ConversationContext = createContext<ConversationContextValue | null>(null)

export function useConversations(): ConversationContextValue {
  const ctx = useContext(ConversationContext)
  if (!ctx) throw new Error('useConversations 必须在 ConversationProvider 内使用')
  return ctx
}

export function ConversationProvider({ children }: { children: ReactNode }) {
  const [conversations, setConversations] = useState<Conversation[]>(loadConversations)
  const [activeId, setActiveId] = useState<string | null>(() => {
    const saved = loadConversations()
    return saved.length > 0 ? saved[0].id : null
  })
  const [streamingConvId, setStreamingConvId] = useState<string | null>(null)

  useEffect(() => {
    saveConversations(conversations)
  }, [conversations])

  const activeConversation = conversations.find((c) => c.id === activeId) ?? null

  /** 就地修改某条消息 —— 全部走函数式更新，避免闭包拿到过期 state */
  const patchMessage = useCallback(
    (convId: string, msgId: string, patch: (m: Message) => Message) => {
      setConversations((prev) =>
        prev.map((c) =>
          c.id !== convId
            ? c
            : {
                ...c,
                messages: c.messages.map((m) => (m.id === msgId ? patch(m) : m)),
                updatedAt: Date.now(),
              },
        ),
      )
    },
    [],
  )

  const createConversation = useCallback(() => {
    const id = Date.now().toString()
    const conv: Conversation = {
      id,
      title: '新对话',
      messages: [],
      createdAt: Date.now(),
      updatedAt: Date.now(),
    }
    setConversations((prev) => [conv, ...prev])
    setActiveId(id)
    return id
  }, [])

  const switchConversation = useCallback((id: string) => setActiveId(id), [])

  const deleteConversation = useCallback((id: string) => {
    setConversations((prev) => {
      const remaining = prev.filter((c) => c.id !== id)
      setActiveId((cur) => (cur === id ? (remaining[0]?.id ?? null) : cur))
      return remaining
    })
  }, [])

  const clearActiveMessages = useCallback(() => {
    setConversations((prev) =>
      prev.map((c) =>
        c.id === activeId
          ? { ...c, messages: [], conversationId: undefined, title: '新对话', updatedAt: Date.now() }
          : c,
      ),
    )
  }, [activeId])

  /**
   * 发送一条提问并流式接收回答。
   *
   * 关键点：整个流程（含流式回调）都跑在 Provider 里，
   * 所以**用户切到其他页面也不会中断**。
   */
  const sendMessage = useCallback(
    async (text: string) => {
      const convId = activeId
      if (!convId || streamingConvId) return

      const conv = conversations.find((c) => c.id === convId)
      const history: ChatHistoryItem[] = (conv?.messages ?? [])
        .slice(-12) // 最近 6 轮
        .filter((m) => m.content.trim())
        .map((m) => ({ role: m.role, content: m.content }))

      const userMsg: Message = { id: `u-${Date.now()}`, role: 'user', content: text }
      const aiMsgId = `a-${Date.now()}`
      const aiMsg: Message = {
        id: aiMsgId,
        role: 'assistant',
        content: '',
        streaming: true,
        liveSteps: [],
      }

      setConversations((prev) =>
        prev.map((c) => {
          if (c.id !== convId) return c
          const messages = [...c.messages, userMsg, aiMsg]
          return {
            ...c,
            messages,
            title: c.messages.length === 0 ? deriveTitle(messages) : c.title,
            updatedAt: Date.now(),
          }
        }),
      )
      setStreamingConvId(convId)

      try {
        await sendChatStream(
          { question: text, conversation_id: conv?.conversationId, history },
          {
            onStep: (label) =>
              patchMessage(convId, aiMsgId, (m) => ({
                ...m,
                liveSteps: [...(m.liveSteps ?? []), label],
              })),

            onToken: (t) =>
              patchMessage(convId, aiMsgId, (m) => ({ ...m, content: m.content + t })),

            onDone: (res) => {
              patchMessage(convId, aiMsgId, (m) => ({
                ...m,
                content: res.answer, // 用收尾后的完整答案覆盖（护栏内容在此补齐）
                sources: res.sources,
                ragSteps: res.rag_steps,
                rewrittenQuestion: res.rewritten_question,
                confidence: res.confidence,
                disclaimer: res.disclaimer,
                streaming: false,
              }))
              // 首轮对话要记住后端返回的 conversation_id
              if (!conv?.conversationId) {
                setConversations((prev) =>
                  prev.map((c) =>
                    c.id === convId ? { ...c, conversationId: res.conversation_id } : c,
                  ),
                )
              }
            },

            onError: (message) =>
              patchMessage(convId, aiMsgId, (m) => ({
                ...m,
                content: `抱歉，请求出错：${message}`,
                streaming: false,
              })),
          },
        )
      } catch (err) {
        patchMessage(convId, aiMsgId, (m) => ({
          ...m,
          content: `抱歉，请求出错：${(err as Error).message}`,
          streaming: false,
        }))
      } finally {
        setStreamingConvId(null)
      }
    },
    [activeId, conversations, streamingConvId, patchMessage],
  )

  return (
    <ConversationContext.Provider
      value={{
        conversations,
        activeId,
        activeConversation,
        isStreaming: streamingConvId !== null,
        createConversation,
        switchConversation,
        deleteConversation,
        clearActiveMessages,
        sendMessage,
      }}
    >
      {children}
    </ConversationContext.Provider>
  )
}
