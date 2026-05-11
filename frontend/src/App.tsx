import { BrowserRouter, Routes, Route, NavLink, useLocation } from 'react-router-dom'
import { Scale, MessageSquare, Database, BarChart3, Settings } from 'lucide-react'
import ChatPage from './pages/ChatPage'
import KnowledgePage from './pages/KnowledgePage'
import EvalPage from './pages/EvalPage'
import SettingsPage from './pages/SettingsPage'

const navItems = [
  { to: '/', icon: MessageSquare, label: '问答' },
  { to: '/knowledge', icon: Database, label: '知识库' },
  { to: '/evaluation', icon: BarChart3, label: '评估' },
  { to: '/settings', icon: Settings, label: '设置' },
]

function AppContent() {
  const location = useLocation()
  const isChat = location.pathname === '/'

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      {/* 顶部导航 */}
      <header className="flex items-center h-14 bg-white border-b border-gray-200 flex-shrink-0 px-4">
        <div className="flex items-center gap-2 mr-8">
          <Scale size={22} className="text-emerald-600" />
          <span className="text-lg font-bold text-gray-800">LaborRAG</span>
          <span className="hidden sm:inline text-sm text-gray-400">劳动法智能问答系统</span>
        </div>
        <nav className="flex items-center gap-1">
          {navItems.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-emerald-50 text-emerald-700'
                    : 'text-gray-500 hover:text-gray-700 hover:bg-gray-100'
                }`
              }
            >
              <Icon size={16} />
              <span className="hidden sm:inline">{label}</span>
            </NavLink>
          ))}
        </nav>
      </header>

      {/* 页面内容 - ChatPage始终挂载，其他页面正常切换 */}
      <main className="flex-1 overflow-hidden relative">
        <div className={`h-full ${isChat ? '' : 'hidden'}`}>
          <ChatPage />
        </div>
        {!isChat && (
          <div className="h-full">
            <Routes>
              <Route path="/knowledge" element={<KnowledgePage />} />
              <Route path="/evaluation" element={<EvalPage />} />
              <Route path="/settings" element={<SettingsPage />} />
            </Routes>
          </div>
        )}
      </main>
    </div>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <AppContent />
    </BrowserRouter>
  )
}
