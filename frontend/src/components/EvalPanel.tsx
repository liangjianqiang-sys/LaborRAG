import { useState } from 'react'
import { runEvaluation, getEvaluationReport, type EvalComparison } from '../api'

export default function EvalPanel() {
  const [running, setRunning] = useState(false)
  const [comparison, setComparison] = useState<EvalComparison | null>(null)
  const [result, setResult] = useState<{ retriever_type: string; scores: Record<string, number> } | null>(null)

  const handleRun = async () => {
    setRunning(true)
    setResult(null)
    try {
      const res = await runEvaluation(5)
      setResult({ retriever_type: res.retriever_type, scores: res.scores })
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
    if (value >= 0.8) return 'text-green-600'
    if (value >= 0.6) return 'text-yellow-600'
    return 'text-red-500'
  }

  return (
    <div className="space-y-4">
      <h3 className="font-semibold text-gray-700 text-sm">RAGAS 评估</h3>

      {/* 运行评估 */}
      <button
        onClick={handleRun}
        disabled={running}
        className="w-full px-3 py-2 bg-emerald-600 text-white text-sm rounded-lg hover:bg-emerald-700 disabled:opacity-50 transition"
      >
        {running ? '评估中...' : '运行评估（5题）'}
      </button>

      {/* 当前评估结果 */}
      {result && (
        <div className="bg-gray-50 rounded-lg p-3 space-y-1">
          <p className="text-xs text-gray-500 font-medium">
            当前策略: <span className="text-emerald-600">{result.retriever_type}</span>
          </p>
          {Object.entries(result.scores).map(([key, val]) => (
            <div key={key} className="flex justify-between text-xs">
              <span className="text-gray-600">{key}</span>
              <span className={`font-mono font-semibold ${scoreColor(val)}`}>
                {val.toFixed(4)}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* 对比报告 */}
      <button
        onClick={handleCompare}
        className="w-full px-3 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700 transition"
      >
        查看对比报告
      </button>

      {comparison && comparison.results && Object.keys(comparison.results).length > 0 && (
        <div className="bg-gray-50 rounded-lg p-3">
          <p className="text-xs text-gray-500 font-medium mb-2">策略对比</p>
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-left py-1 text-gray-500">策略</th>
                {comparison.metrics.map((m) => (
                  <th key={m} className="text-right py-1 text-gray-500">{m}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Object.entries(comparison.results).map(([type, scores]) => (
                <tr key={type} className="border-b border-gray-100">
                  <td className="py-1 text-emerald-600 font-medium">{type}</td>
                  {comparison.metrics.map((m) => (
                    <td key={m} className={`text-right py-1 font-mono ${scoreColor(scores[m] || 0)}`}>
                      {(scores[m] || 0).toFixed(4)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {comparison?.message && (
        <p className="text-xs text-gray-400 text-center">{comparison.message}</p>
      )}
    </div>
  )
}
