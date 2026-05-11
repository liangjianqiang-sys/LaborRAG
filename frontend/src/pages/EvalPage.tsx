import { useState } from 'react'
import { BarChart3, Play, FileBarChart, Info } from 'lucide-react'
import { runEvaluation, getEvaluationReport, type EvalComparison } from '../api'

const STRATEGY_LABELS: Record<string, string> = {
  vector: 'V1 纯向量',
  hybrid: 'V2 混合检索',
  reranked: 'V2 重排序',
  crag: 'V3 CRAG纠错',
}

export default function EvalPage() {
  const [running, setRunning] = useState(false)
  const [comparison, setComparison] = useState<EvalComparison | null>(null)
  const [result, setResult] = useState<{ retriever_type: string; scores: Record<string, number>; sample_count: number } | null>(null)
  const [sampleCount, setSampleCount] = useState(5)

  const handleRun = async () => {
    setRunning(true)
    setResult(null)
    try {
      const res = await runEvaluation(sampleCount)
      setResult({ retriever_type: res.retriever_type, scores: res.scores, sample_count: res.sample_count })
    } catch (err) {
      alert(`评估失败: ${(err as Error).message}`)
    } finally {
      setRunning(false)
    }
  }

  const handleCompare = async () => {
    try {
      const report = await getEvaluationReport()
      setComparison(report)
    } catch (err) {
      alert(`获取报告失败: ${(err as Error).message}`)
    }
  }

  const scoreColor = (value: number) => {
    if (value >= 0.8) return 'text-green-600 bg-green-50'
    if (value >= 0.6) return 'text-yellow-600 bg-yellow-50'
    return 'text-red-500 bg-red-50'
  }

  const scoreBar = (value: number) => {
    const pct = Math.round(value * 100)
    const color = value >= 0.8 ? 'bg-green-500' : value >= 0.6 ? 'bg-yellow-500' : 'bg-red-500'
    return (
      <div className="flex items-center gap-2">
        <div className="flex-1 h-2 bg-gray-200 rounded-full overflow-hidden">
          <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
        </div>
        <span className={`text-sm font-mono font-semibold px-2 py-0.5 rounded ${scoreColor(value)}`}>
          {value.toFixed(4)}
        </span>
      </div>
    )
  }

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-6">
      <h2 className="text-2xl font-bold text-gray-800 flex items-center gap-2">
        <BarChart3 size={24} className="text-blue-600" />
        RAGAS 评估
      </h2>

      {/* 运行评估 */}
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <h3 className="font-semibold text-gray-700 mb-4 flex items-center gap-2">
          <Play size={18} className="text-emerald-600" />
          运行评估
        </h3>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <label className="text-sm text-gray-600">样本数量:</label>
            <select
              value={sampleCount}
              onChange={(e) => setSampleCount(Number(e.target.value))}
              className="rounded-lg border border-gray-300 px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500"
            >
              <option value={3}>3</option>
              <option value={5}>5</option>
              <option value={10}>10</option>
              <option value={20}>全部(20)</option>
            </select>
          </div>
          <button
            onClick={handleRun}
            disabled={running}
            className="flex items-center gap-2 rounded-lg bg-emerald-600 px-4 py-2 text-sm text-white hover:bg-emerald-700 disabled:opacity-50 transition"
          >
            {running ? (
              <>
                <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                评估中...
              </>
            ) : (
              <>
                <Play size={16} />
                运行评估
              </>
            )}
          </button>
        </div>

        {/* 当前评估结果 */}
        {result && (
          <div className="mt-6 space-y-3">
            <div className="flex items-center justify-between">
              <p className="text-sm text-gray-500">
                当前策略: <span className="text-emerald-600 font-semibold">{STRATEGY_LABELS[result.retriever_type] || result.retriever_type}</span>
              </p>
              <p className="text-xs text-gray-400">{result.sample_count} 个样本</p>
            </div>
            <div className="space-y-2">
              {Object.entries(result.scores).map(([key, val]) => (
                <div key={key}>
                  <p className="text-xs text-gray-500 mb-1">{key}</p>
                  {scoreBar(val)}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* 对比报告 */}
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-semibold text-gray-700 flex items-center gap-2">
            <FileBarChart size={18} className="text-blue-600" />
            策略对比报告
          </h3>
          <button
            onClick={handleCompare}
            className="flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-sm text-white hover:bg-blue-700 transition"
          >
            <FileBarChart size={16} />
            查看对比
          </button>
        </div>

        <div className="flex items-start gap-2 bg-blue-50 border border-blue-100 rounded-lg p-3 mb-4">
          <Info size={16} className="text-blue-500 flex-shrink-0 mt-0.5" />
          <div className="text-xs text-blue-700">
            <p className="font-medium mb-1">V3 对比实验方法</p>
            <p>在后端运行 <code className="bg-blue-100 px-1 rounded">python run_v3_comparison.py</code> 可自动执行 V1/V2/V3 三组对比实验。</p>
            <p className="mt-1">也可指定 <code className="bg-blue-100 px-1 rounded">--only crag</code> 只运行V3，<code className="bg-blue-100 px-1 rounded">--sample 5</code> 限制样本数。</p>
          </div>
        </div>

        {comparison && comparison.results && Object.keys(comparison.results).length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b-2 border-gray-200">
                  <th className="text-left py-3 px-3 text-gray-600 font-semibold">策略</th>
                  {comparison.metrics.map((m) => (
                    <th key={m} className="text-center py-3 px-3 text-gray-600 font-semibold">{m}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(comparison.results).map(([type, scores]) => (
                  <tr key={type} className="border-b border-gray-100 hover:bg-gray-50">
                    <td className="py-3 px-3 font-semibold text-emerald-600">{STRATEGY_LABELS[type] || type}</td>
                    {comparison.metrics.map((m) => {
                      const val = scores[m] || 0
                      return (
                        <td key={m} className="py-3 px-3 text-center">
                          <span className={`inline-block px-2 py-1 rounded font-mono font-semibold text-xs ${scoreColor(val)}`}>
                            {val.toFixed(4)}
                          </span>
                        </td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center text-gray-400 text-sm py-8">
            暂无对比数据，请先运行不同策略的评估
          </div>
        )}

        {comparison?.message && (
          <p className="text-xs text-gray-400 text-center mt-3">{comparison.message}</p>
        )}
      </div>
    </div>
  )
}
