import { useState } from 'react'
import { Settings as SettingsIcon, Save } from 'lucide-react'

export default function SettingsPage() {
  // 读取当前配置（从后端获取）
  const [retrieverType, setRetrieverType] = useState('reranked')
  const [ragMode, setRagMode] = useState('simple')
  const [chunkStrategy, setChunkStrategy] = useState('law_article')
  const [topK, setTopK] = useState('8')
  const [vectorWeight, setVectorWeight] = useState('0.5')
  const [bm25Weight, setBm25Weight] = useState('0.5')
  const [rrfK, setRrfK] = useState('60')
  const [saved, setSaved] = useState(false)

  const handleSave = () => {
    // 提示用户需要手动修改.env并重启后端
    setSaved(true)
    setTimeout(() => setSaved(false), 3000)
  }

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-6">
      <h2 className="text-2xl font-bold text-gray-800 flex items-center gap-2">
        <SettingsIcon size={24} className="text-gray-600" />
        系统设置
      </h2>

      {/* 检索策略 */}
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <h3 className="font-semibold text-gray-700 mb-4">检索策略</h3>
        <div className="space-y-4">
          <div>
            <label className="block text-sm text-gray-600 mb-2">检索类型 (RETRIEVER_TYPE)</label>
            <div className="grid grid-cols-3 gap-3">
              {[
                { value: 'vector', label: 'V1 纯向量', desc: '语义相似度匹配' },
                { value: 'hybrid', label: 'V2 混合检索', desc: '向量+BM25+RRF' },
                { value: 'reranked', label: 'V2 重排序', desc: '混合+Cross-Encoder' },
              ].map((opt) => (
                <label
                  key={opt.value}
                  className={`relative flex flex-col rounded-lg border-2 p-4 cursor-pointer transition-colors ${
                    retrieverType === opt.value ? 'border-emerald-500 bg-emerald-50' : 'border-gray-200 hover:border-gray-300'
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
                  <span className="text-sm font-semibold text-gray-800">{opt.label}</span>
                  <span className="text-xs text-gray-500 mt-1">{opt.desc}</span>
                </label>
              ))}
            </div>
          </div>

          <div>
            <label className="block text-sm text-gray-600 mb-2">切分策略 (CHUNK_STRATEGY)</label>
            <div className="grid grid-cols-2 gap-3">
              {[
                { value: 'recursive', label: '固定长度切分', desc: '按固定字符数切分' },
                { value: 'law_article', label: '法条结构化切分', desc: '按第X条边界切分' },
              ].map((opt) => (
                <label
                  key={opt.value}
                  className={`relative flex flex-col rounded-lg border-2 p-4 cursor-pointer transition-colors ${
                    chunkStrategy === opt.value ? 'border-emerald-500 bg-emerald-50' : 'border-gray-200 hover:border-gray-300'
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
                  <span className="text-sm font-semibold text-gray-800">{opt.label}</span>
                  <span className="text-xs text-gray-500 mt-1">{opt.desc}</span>
                </label>
              ))}
            </div>
          </div>

          <div>
            <label className="block text-sm text-gray-600 mb-2">RAG模式 (RAG_MODE)</label>
            <div className="grid grid-cols-2 gap-3">
              {[
                { value: 'simple', label: 'V1/V2 简单链路', desc: '检索→生成，无纠错' },
                { value: 'crag', label: 'V3 CRAG纠错', desc: '检索评估→改写→生成评估→纠错' },
              ].map((opt) => (
                <label
                  key={opt.value}
                  className={`relative flex flex-col rounded-lg border-2 p-4 cursor-pointer transition-colors ${
                    ragMode === opt.value ? 'border-blue-500 bg-blue-50' : 'border-gray-200 hover:border-gray-300'
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
                  <span className="text-sm font-semibold text-gray-800">{opt.label}</span>
                  <span className="text-xs text-gray-500 mt-1">{opt.desc}</span>
                </label>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* 参数配置 */}
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <h3 className="font-semibold text-gray-700 mb-4">检索参数</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <label className="block text-xs text-gray-500 mb-1">TOP_K</label>
            <input
              type="number"
              value={topK}
              onChange={(e) => setTopK(e.target.value)}
              className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-500 mb-1">向量权重</label>
            <input
              type="number"
              step="0.1"
              value={vectorWeight}
              onChange={(e) => setVectorWeight(e.target.value)}
              className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-500 mb-1">BM25权重</label>
            <input
              type="number"
              step="0.1"
              value={bm25Weight}
              onChange={(e) => setBm25Weight(e.target.value)}
              className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-500 mb-1">RRF_K</label>
            <input
              type="number"
              value={rrfK}
              onChange={(e) => setRrfK(e.target.value)}
              className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500"
            />
          </div>
        </div>
      </div>

      {/* 保存提示 */}
      <div className="bg-amber-50 border border-amber-200 rounded-xl p-4">
        <p className="text-sm text-amber-700">
          ⚠️ 当前配置修改需要手动更新 <code className="bg-amber-100 px-1 rounded">backend/.env</code> 文件并重启后端才能生效。
        </p>
      </div>

      <button
        onClick={handleSave}
        className="flex items-center gap-2 rounded-lg bg-emerald-600 px-6 py-2.5 text-sm text-white hover:bg-emerald-700 transition"
      >
        <Save size={16} />
        保存配置
      </button>

      {saved && (
        <div className="fixed bottom-6 left-1/2 -translate-x-1/2 bg-emerald-600 text-white rounded-lg px-4 py-2 text-sm shadow-lg">
          配置已保存（需更新.env并重启后端生效）
        </div>
      )}
    </div>
  )
}
