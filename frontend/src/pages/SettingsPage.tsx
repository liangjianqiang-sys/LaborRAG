import { useState } from 'react'
import { Settings as SettingsIcon, Save, SlidersHorizontal, AlertTriangle, Gauge, Layers, ArrowLeftRight, SplitSquareHorizontal } from 'lucide-react'

export default function SettingsPage() {
  const [retrieverType, setRetrieverType] = useState('reranked')
  const [ragMode, setRagMode] = useState('simple')
  const [chunkStrategy, setChunkStrategy] = useState('law_article')
  const [topK, setTopK] = useState('8')
  const [vectorWeight, setVectorWeight] = useState('0.5')
  const [bm25Weight, setBm25Weight] = useState('0.5')
  const [rrfK, setRrfK] = useState('60')
  const [saved, setSaved] = useState(false)

  const handleSave = () => {
    setSaved(true)
    setTimeout(() => setSaved(false), 3000)
  }

  return (
    <div className="max-w-4xl mx-auto p-5 lg:p-6 space-y-5 overflow-y-auto h-full">
      {/* ── 页面标题 ── */}
      <div className="flex items-center gap-3">
        <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-brand-500 to-accent-500 flex items-center justify-center shadow-sm shadow-brand-200/40">
          <SettingsIcon size={20} className="text-white" />
        </div>
        <div>
          <h2 className="text-lg font-bold text-gradient-brand">系统设置</h2>
          <p className="text-xs text-slate-400 mt-0.5">配置检索策略与 RAG 链路参数</p>
        </div>
      </div>

      {/* ── 检索策略 ── */}
      <div className="bg-white rounded-xl border border-slate-200/80 p-5 shadow-card">
        <div className="flex items-center gap-2 mb-4 pb-3 border-b border-brand-100/50">
          <SlidersHorizontal size={16} className="text-brand-500" />
          <h3 className="font-semibold text-slate-700 text-sm">检索策略</h3>
        </div>

        <div className="space-y-5">
          {/* 检索类型 */}
          <div>
            <label className="flex items-center gap-1.5 text-xs font-medium text-slate-500 mb-2.5 uppercase tracking-wider">
              <Gauge size={14} className="text-brand-400" />
              检索类型
            </label>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
              {[
                { value: 'vector', label: 'V1 纯向量', desc: '语义相似度匹配', icon: Layers },
                { value: 'hybrid', label: 'V2 混合检索', desc: '向量 + BM25 + RRF 融合', icon: ArrowLeftRight },
                { value: 'reranked', label: 'V2 重排序', desc: '混合检索 + Cross-Encoder 重排', icon: Gauge },
              ].map((opt) => (
                <label
                  key={opt.value}
                  className={`relative flex items-start gap-3 rounded-xl border-2 p-4 cursor-pointer transition-all duration-200 ${
                    retrieverType === opt.value
                      ? 'border-brand-500 bg-brand-50/60 shadow-sm shadow-brand-200/30'
                      : 'border-slate-100 bg-white hover:border-brand-200 hover:bg-brand-50/30'
                  }`}
                >
                  <input
                    type="radio"
                    name="retriever"
                    value={opt.value}
                    checked={retrieverType === opt.value}
                    onChange={(e) => setRetrieverType(e.target.value)}
                    className="sr-only"
                  />
                  <div className={`flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center ${
                    retrieverType === opt.value ? 'bg-brand-100 text-brand-600' : 'bg-slate-100 text-slate-400'
                  } transition-colors duration-200`}>
                    <opt.icon size={16} />
                  </div>
                  <div>
                    <span className={`text-sm font-semibold ${
                      retrieverType === opt.value ? 'text-brand-700' : 'text-slate-700'
                    }`}>
                      {opt.label}
                    </span>
                    <p className="text-[11px] text-slate-400 mt-0.5 leading-relaxed">{opt.desc}</p>
                  </div>
                  {retrieverType === opt.value && (
                    <div className="absolute top-2 right-2 w-2.5 h-2.5 rounded-full bg-brand-500" />
                  )}
                </label>
              ))}
            </div>
          </div>

          {/* 切分策略 */}
          <div>
            <label className="flex items-center gap-1.5 text-xs font-medium text-slate-500 mb-2.5 uppercase tracking-wider">
              <SplitSquareHorizontal size={14} className="text-accent-400" />
              切分策略
            </label>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
              {[
                { value: 'recursive', label: '固定长度切分', desc: '按固定字符数切分文本块', icon: Layers },
                { value: 'law_article', label: '法条结构化切分', desc: '按"第X条"边界切分，保留法条完整性', icon: SplitSquareHorizontal },
              ].map((opt) => (
                <label
                  key={opt.value}
                  className={`relative flex items-start gap-3 rounded-xl border-2 p-4 cursor-pointer transition-all duration-200 ${
                    chunkStrategy === opt.value
                      ? 'border-accent-500 bg-accent-50/60 shadow-sm shadow-accent-200/30'
                      : 'border-slate-100 bg-white hover:border-accent-200 hover:bg-accent-50/30'
                  }`}
                >
                  <input
                    type="radio"
                    name="chunk"
                    value={opt.value}
                    checked={chunkStrategy === opt.value}
                    onChange={(e) => setChunkStrategy(e.target.value)}
                    className="sr-only"
                  />
                  <div className={`flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center ${
                    chunkStrategy === opt.value ? 'bg-accent-100 text-accent-600' : 'bg-slate-100 text-slate-400'
                  } transition-colors duration-200`}>
                    <opt.icon size={16} />
                  </div>
                  <div>
                    <span className={`text-sm font-semibold ${
                      chunkStrategy === opt.value ? 'text-accent-700' : 'text-slate-700'
                    }`}>
                      {opt.label}
                    </span>
                    <p className="text-[11px] text-slate-400 mt-0.5 leading-relaxed">{opt.desc}</p>
                  </div>
                  {chunkStrategy === opt.value && (
                    <div className="absolute top-2 right-2 w-2.5 h-2.5 rounded-full bg-accent-500" />
                  )}
                </label>
              ))}
            </div>
          </div>

          {/* RAG 模式 */}
          <div>
            <label className="flex items-center gap-1.5 text-xs font-medium text-slate-500 mb-2.5 uppercase tracking-wider">
              <ArrowLeftRight size={14} className="text-cyan-500" />
              RAG 模式
            </label>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
              {[
                { value: 'simple', label: 'V1/V2 简单链路', desc: '检索 → 生成（快速，无纠错机制）', icon: Layers },
                { value: 'crag', label: 'V3 CRAG 纠错', desc: '检索评估 → 改写 → 生成 → 回答评估 → 纠错', icon: ArrowLeftRight },
              ].map((opt) => (
                <label
                  key={opt.value}
                  className={`relative flex items-start gap-3 rounded-xl border-2 p-4 cursor-pointer transition-all duration-200 ${
                    ragMode === opt.value
                      ? 'border-cyan-500 bg-cyan-50/60 shadow-sm shadow-cyan-200/30'
                      : 'border-slate-100 bg-white hover:border-cyan-200 hover:bg-cyan-50/30'
                  }`}
                >
                  <input
                    type="radio"
                    name="ragmode"
                    value={opt.value}
                    checked={ragMode === opt.value}
                    onChange={(e) => setRagMode(e.target.value)}
                    className="sr-only"
                  />
                  <div className={`flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center ${
                    ragMode === opt.value ? 'bg-cyan-100 text-cyan-600' : 'bg-slate-100 text-slate-400'
                  } transition-colors duration-200`}>
                    <opt.icon size={16} />
                  </div>
                  <div>
                    <span className={`text-sm font-semibold ${
                      ragMode === opt.value ? 'text-cyan-700' : 'text-slate-700'
                    }`}>
                      {opt.label}
                    </span>
                    <p className="text-[11px] text-slate-400 mt-0.5 leading-relaxed">{opt.desc}</p>
                  </div>
                  {ragMode === opt.value && (
                    <div className="absolute top-2 right-2 w-2.5 h-2.5 rounded-full bg-cyan-500" />
                  )}
                </label>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* ── 检索参数 ── */}
      <div className="bg-white rounded-xl border border-slate-200/80 p-5 shadow-card">
        <div className="flex items-center gap-2 mb-4 pb-3 border-b border-brand-100/50">
          <Gauge size={16} className="text-brand-500" />
          <h3 className="font-semibold text-slate-700 text-sm">检索参数</h3>
        </div>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          {[
            { label: 'TOP_K', value: topK, setter: setTopK, desc: '检索返回数量' },
            { label: '向量权重', value: vectorWeight, setter: setVectorWeight, desc: '混合检索向量权重', step: '0.1' },
            { label: 'BM25 权重', value: bm25Weight, setter: setBm25Weight, desc: '混合检索 BM25 权重', step: '0.1' },
            { label: 'RRF_K', value: rrfK, setter: setRrfK, desc: 'RRF 融合参数' },
          ].map((param) => (
            <div key={param.label}>
              <label className="block text-xs font-medium text-slate-500 mb-1.5">{param.label}</label>
              <input
                type="number"
                step={param.step || '1'}
                value={param.value}
                onChange={(e) => param.setter(e.target.value)}
                className="w-full rounded-lg border border-slate-200 bg-slate-50/50 px-3 py-2 text-sm text-slate-700 focus:outline-none focus:ring-2 focus:ring-brand-400/30 focus:border-brand-400 focus:bg-white transition-all duration-200"
              />
              <p className="text-[10px] text-slate-400 mt-1">{param.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* ── 提示 ── */}
      <div className="flex items-start gap-3 rounded-xl border border-amber-200/70 bg-gradient-to-r from-amber-50 to-brand-50/30 p-4 shadow-sm">
        <AlertTriangle size={18} className="text-amber-500 flex-shrink-0 mt-0.5" />
        <div>
          <p className="text-sm font-medium text-amber-800">需要手动生效</p>
          <p className="text-xs text-amber-600/80 mt-1 leading-relaxed">
            当前配置修改后需要手动更新 <code className="bg-brand-50 px-1.5 py-0.5 rounded text-[11px] font-mono text-brand-600 border border-brand-200/60">backend/.env</code> 文件并重启后端才能生效。
          </p>
        </div>
      </div>

      {/* ── 保存按钮 ── */}
      <div className="flex justify-end">
        <button
          onClick={handleSave}
          className="inline-flex items-center gap-2 rounded-lg btn-brand px-6 py-2.5 text-sm transition-all duration-200"
        >
          <Save size={16} />
          保存配置
        </button>
      </div>

      {/* ── Toast ── */}
      {saved && (
        <div className="fixed bottom-6 left-1/2 -translate-x-1/2 bg-gradient-to-r from-brand-500 to-accent-500 text-white rounded-xl px-5 py-2.5 text-sm font-medium shadow-toast animate-fade-in z-50">
          配置已保存（需更新 .env 并重启后端生效）
        </div>
      )}
    </div>
  )
}
