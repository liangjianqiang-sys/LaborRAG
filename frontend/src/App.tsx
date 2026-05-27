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
    <div className="flex flex-col h-screen">
      {/* ── 顶部导航 ── */}
      <header className="flex items-center h-14 bg-white/80 backdrop-blur-md border-b border-brand-100/50 flex-shrink-0 px-4 lg:px-6 shadow-[0_1px_3px_rgba(59,130,246,0.04)]">
        <div className="flex items-center gap-2.5 mr-6 lg:mr-10">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-brand-500 to-accent-500 flex items-center justify-center shadow-[0_2px_8px_rgba(59,130,246,0.25)]">
            <Scale size={18} className="text-white" strokeWidth={2.2} />
          </div>
          <span className="text-base font-bold text-gradient-brand tracking-tight">LaborRAG</span>
          <span className="hidden sm:inline text-sm text-slate-400 font-normal border-l border-slate-200 pl-3 ml-0.5">
            劳动法智能问答系统
          </span>
        </div>
        <nav className="flex items-center gap-0.5">
          {navItems.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-all duration-200 ${
                  isActive
                    ? 'bg-brand-50 text-brand-700 shadow-sm shadow-brand-200/50'
                    : 'text-slate-500 hover:text-brand-600 hover:bg-brand-50/50'
                }`
              }
            >
              <Icon size={16} strokeWidth={1.8} />
              <span className="hidden sm:inline">{label}</span>
            </NavLink>
          ))}
        </nav>
      </header>

      {/* ── 页面内容 ── */}
      <main className="flex-1 overflow-hidden relative">
        <div className={`h-full transition-opacity duration-200 ${isChat ? 'opacity-100' : 'opacity-100'}`}>
          {isChat ? (
            <ChatPage />
          ) : (
            <div className="h-full animate-fade-in">
              <Routes>
                <Route path="/knowledge" element={<KnowledgePage />} />
                <Route path="/evaluation" element={<EvalPage />} />
                <Route path="/settings" element={<SettingsPage />} />
              </Routes>
            </div>
          )}
        </div>
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
