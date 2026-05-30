import { useState, useEffect, useCallback } from 'react'

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: { content: string; source: string; score: number; page?: number }[]
  cragSteps?: string[]
  rewrittenQuestion?: string
  ragMode?: string
  confidence?: number
  disclaimer?: boolean
}

export interface Conversation {
  id: string
  title: string
  messages: Message[]
  conversationId?: string  // 后端返回的 conversation_id
  createdAt: number
  updatedAt: number
}

const STORAGE_KEY = 'laborrag_conversations'

function loadConversations(): Conversation[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? JSON.parse(raw) : []
  } catch {
    return []
  }
}

function saveConversations(conversations: Conversation[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations))
}

/** 从消息中自动提取对话标题（取第一条用户消息的前20字） */
function deriveTitle(messages: Message[]): string {
  const first = messages.find((m) => m.role === 'user')
  if (!first) return '新对话'
  const text = first.content.trim()
  return text.length > 20 ? text.slice(0, 20) + '…' : text
}

export function useConversations() {
  const [conversations, setConversations] = useState<Conversation[]>(loadConversations)
  const [activeId, setActiveId] = useState<string | null>(() => {
    const saved = loadConversations()
    return saved.length > 0 ? saved[0].id : null
  })

  // 每次变化都持久化
  useEffect(() => {
    saveConversations(conversations)
  }, [conversations])

  const activeConversation = conversations.find((c) => c.id === activeId) ?? null

  /** 新建对话 */
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

  /** 切换对话 */
  const switchConversation = useCallback((id: string) => {
    setActiveId(id)
  }, [])

  /** 删除对话 */
  const deleteConversation = useCallback(
    (id: string) => {
      setConversations((prev) => {
        const remaining = prev.filter((c) => c.id !== id)
        // 如果删除的是当前活跃对话，切换到剩余的第一个
        if (activeId === id) {
          setActiveId(remaining.length > 0 ? remaining[0].id : null)
        }
        return remaining
      })
    },
    [activeId],
  )

  /** 追加消息到当前对话 */
  const addMessage = useCallback(
    (message: Message) => {
      if (!activeId) return
      setConversations((prev) =>
        prev.map((c) => {
          if (c.id !== activeId) return c
          const newMessages = [...c.messages, message]
          return {
            ...c,
            messages: newMessages,
            title: c.messages.length === 0 ? deriveTitle(newMessages) : c.title,
            conversationId:
              message.role === 'assistant' && message.ragMode
                ? message.id  // 使用后端返回的 conversation_id
                : c.conversationId,
            updatedAt: Date.now(),
          }
        }),
      )
    },
    [activeId],
  )

  /** 更新当前对话的后端 conversation_id */
  const setBackendConversationId = useCallback(
    (backendId: string) => {
      if (!activeId) return
      setConversations((prev) =>
        prev.map((c) =>
          c.id === activeId ? { ...c, conversationId: backendId, updatedAt: Date.now() } : c,
        ),
      )
    },
    [activeId],
  )

  /** 清空当前对话消息（用于新建对话时重置） */
  const clearActiveMessages = useCallback(() => {
    if (!activeId) return
    setConversations((prev) =>
      prev.map((c) =>
        c.id === activeId
          ? { ...c, messages: [], conversationId: undefined, title: '新对话', updatedAt: Date.now() }
          : c,
      ),
    )
  }, [activeId])

  return {
    conversations,
    activeId,
    activeConversation,
    createConversation,
    switchConversation,
    deleteConversation,
    addMessage,
    setBackendConversationId,
    clearActiveMessages,
  }
}
