import { useState, useEffect } from 'react'
import ChatWindow from '../components/ChatWindow'
import ConversationSidebar from '../components/ConversationSidebar'
import { useConversations } from '../hooks/ConversationContext'

export default function ChatPage() {
  const {
    conversations,
    activeId,
    activeConversation,
    isStreaming,
    createConversation,
    switchConversation,
    deleteConversation,
    sendMessage,
  } = useConversations()

  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)

  // 首次访问时自动创建一个对话
  useEffect(() => {
    if (conversations.length === 0) {
      createConversation()
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const handleNewChat = () => {
    createConversation()
  }

  return (
    <div className="flex h-full">
      {/* ── 侧边栏 ── */}
      <ConversationSidebar
        conversations={conversations}
        activeId={activeId}
        onSelect={switchConversation}
        onNew={handleNewChat}
        onDelete={deleteConversation}
        collapsed={sidebarCollapsed}
        onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
      />

      {/* ── 聊天区域 ── */}
      <div className="flex-1 min-w-0">
        <ChatWindow
          key={activeId}
          conversation={activeConversation}
          isStreaming={isStreaming}
          onSend={sendMessage}
          onNewChat={handleNewChat}
        />
      </div>
    </div>
  )
}
