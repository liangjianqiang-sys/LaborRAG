import { useState, useEffect, useRef } from 'react'
import { BarChart3, Play, FileBarChart, Info, ChevronDown, ChevronUp, Search, Calculator, GitCompare, Eye, AlertTriangle, Target, Shield, Activity, RotateCcw, FastForward, RefreshCw, ListChecks } from 'lucide-react'
import { startEvaluation, getEvaluationStatus, getEvaluationReport, getObservability, getBadCases, createPersistentTask, startPersistentTask, resumePersistentTask, restartPersistentTask, retryFailedPersistentTask, listPersistentTasks, getPersistentProgress, getPersistentReport, type EvalResult, type EvalComparison, type Observability, type BadCase, type PersistentTask, type PersistentProgress, type PersistentReport } from '../api'

const STRATEGY_LABELS: Record<string, string> = {
  vector: 'V1 纯向量',
  hybrid: 'V2 混合检索',
  reranked: 'V2 重排序',
  crag: 'V3 CRAG纠错',
  simple: 'V1/V2 简单链路',
  agent: 'V4 Agentic RAG',
}

const TYPE_LABELS: Record<string, string> = {
  retrieve: '法条查询',
  calculate: '计算类',
  compare: '对比类',
}

const TYPE_ICONS: Record<string, typeof Search> = {
  retrieve: Search,
  calculate: Calculator,
  compare: GitCompare,
}

const POLL_INTERVAL = 3000

export default function EvalPage() {
  const [running, setRunning] = useState(false)
  const [comparison, setComparison] = useState<EvalComparison | null>(null)
  const [result, setResult] = useState<EvalResult | null>(null)
  const [observability, setObservability] = useState<Observability | null>(null)
  const [badCases, setBadCases] = useState<BadCase[]>([])
  const [sampleCount, setSampleCount] = useState(5)
  const [ragMode, setRagMode] = useState<string>('')
  const [questionType, setQuestionType] = useState<string>('')
  const [expandedDetail, setExpandedDetail] = useState<number | null>(null)
  const [expandedBadCase, setExpandedBadCase] = useState<number | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // 断点续评状态
  const [pTasks, setPTasks] = useState<PersistentTask[]>([])
  const [pProgress, setPProgress] = useState<PersistentProgress | null>(null)
  const [pReport, setPReport] = useState<PersistentReport | null>(null)
  const [pRunning, setPRunning] = useState(false)
  const pPollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // 断点续评轮询
  const startPPolling = (taskId: string) => {
    if (pPollRef.current) clearInterval(pPollRef.current)
    pPollRef.current = setInterval(async () => {
      try {
        const prog = await getPersistentProgress(taskId)
        setPProgress(prog)
        if (prog.status !== 'running') {
          setPRunning(false)
          if (pPollRef.current) clearInterval(pPollRef.current)
          pPollRef.current = null
          listPersistentTasks().then(r => setPTasks(r.tasks)).catch(() => {})
        }
      } catch {
        if (pPollRef.current) clearInterval(pPollRef.current)
        pPollRef.current = null
        setPRunning(false)
      }
    }, POLL_INTERVAL)
  }

  // 轮询评估状态
  const startPolling = () => {
    if (pollRef.current) return
    pollRef.current = setInterval(async () => {
      try {
        const status = await getEvaluationStatus()
        setRunning(status.running)
        if (status.error && !status.running) {
          alert(`评估失败: ${status.error}`)
          stopPolling()
        } else if (status.result && !status.running) {
          setResult(status.result)
          stopPolling()
        }
      } catch {
        stopPolling()
      }
    }, POLL_INTERVAL)
  }

  const stopPolling = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }

  // 页面加载时检查是否有正在运行的评估
  useEffect(() => {
    getEvaluationStatus().then(status => {
      if (status.running) {
        setRunning(true)
        startPolling()
      } else if (status.result) {
        setResult(status.result)
      }
    }).catch(() => {})
    // 加载断点续评任务列表
    listPersistentTasks().then(r => setPTasks(r.tasks)).catch(() => {})
    return () => { stopPolling(); if (pPollRef.current) clearInterval(pPollRef.current) }
  }, [])

  const handleRun = async () => {
    try {
      await startEvaluation(sampleCount, ragMode || undefined, questionType || undefined)
      setRunning(true)
      setResult(null)
      startPolling()
    } catch (err) {
      alert(`启动评估失败: ${(err as Error).message}`)
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
    if (value >= 0.4) return 'text-orange-600 bg-orange-50'
    return 'text-red-500 bg-red-50'
  }

  const scoreBar = (value: number) => {
    const pct = Math.round(value * 100)
    const color = value >= 0.8 ? 'bg-green-500' : value >= 0.6 ? 'bg-yellow-500' : value >= 0.4 ? 'bg-orange-500' : 'bg-red-500'
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

  const metricLabel = (key: string) => {
    const labels: Record<string, string> = {
      faithfulness: '忠实度',
      answer_relevancy: '答案相关性',
      context_precision: '上下文精确度',
      context_recall: '上下文召回率',
    }
    return labels[key] || key
  }

  return (
    <div className="max-w-5xl mx-auto p-6 space-y-6">
      <h2 className="text-2xl font-bold text-gray-800 flex items-center gap-2">
        <BarChart3 size={24} className="text-blue-600" />
        评估系统
      </h2>

      {/* 运行评估 */}
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <h3 className="font-semibold text-gray-700 mb-4 flex items-center gap-2">
          <Play size={18} className="text-emerald-600" />
          运行评估
        </h3>
        <div className="flex flex-wrap items-center gap-4">
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
              <option value={20}>全部(24)</option>
            </select>
          </div>
          <div className="flex items-center gap-2">
            <label className="text-sm text-gray-600">RAG模式:</label>
            <select
              value={ragMode}
              onChange={(e) => setRagMode(e.target.value)}
              className="rounded-lg border border-gray-300 px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500"
            >
              <option value="">当前配置</option>
              <option value="simple">V1/V2 简单链路</option>
              <option value="crag">V3 CRAG纠错</option>
              <option value="agent">V4 Agentic RAG</option>
            </select>
          </div>
          <div className="flex items-center gap-2">
            <label className="text-sm text-gray-600">问题类型:</label>
            <select
              value={questionType}
              onChange={(e) => setQuestionType(e.target.value)}
              className="rounded-lg border border-gray-300 px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500"
            >
              <option value="">全部类型</option>
              <option value="retrieve">法条查询</option>
              <option value="calculate">计算类</option>
              <option value="compare">对比类</option>
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
          <div className="mt-6 space-y-4">
            <div className="flex items-center justify-between">
              <p className="text-sm text-gray-500">
                当前策略: <span className="text-emerald-600 font-semibold">{STRATEGY_LABELS[result.rag_mode] || result.rag_mode}</span>
              </p>
              <p className="text-xs text-gray-400">{result.sample_count} 个样本</p>
            </div>

            {/* 三元组雷达图 */}
            {result.triad && (
              <div className="space-y-2">
                <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider flex items-center gap-1">
                  <Target size={12} /> 三元组核心指标
                </p>
                <div className="grid grid-cols-3 gap-3">
                  {[
                    { key: 'context_relevancy', label: '上下文相关性', icon: Search, desc: '检索质量' },
                    { key: 'faithfulness', label: '忠实度', icon: Shield, desc: '生成质量' },
                    { key: 'answer_relevancy', label: '答案相关性', icon: Activity, desc: '端到端' },
                  ].map(({ key, label, icon: Icon, desc }) => {
                    const val = result.triad[key as keyof typeof result.triad] ?? 0
                    return (
                      <div key={key} className="border border-gray-100 rounded-lg p-3 text-center">
                        <Icon size={16} className="mx-auto mb-1 text-blue-500" />
                        <p className="text-xs text-gray-500">{label}</p>
                        <p className="text-[10px] text-gray-400">{desc}</p>
                        <span className={`inline-block mt-1 text-lg font-bold font-mono px-2 py-0.5 rounded ${scoreColor(val)}`}>
                          {(val * 100).toFixed(1)}%
                        </span>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}

            {/* 检索指标卡片 */}
            {result.retrieval && (
              <div className="space-y-2">
                <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider flex items-center gap-1">
                  <Search size={12} /> 检索指标
                </p>
                <div className="grid grid-cols-5 gap-2">
                  {[
                    { key: 'precision@5', label: 'P@5' },
                    { key: 'recall@5', label: 'R@5' },
                    { key: 'f1@5', label: 'F1@5' },
                    { key: 'mrr', label: 'MRR' },
                    { key: 'map', label: 'MAP' },
                  ].map(({ key, label }) => {
                    const val = (result.retrieval as unknown as Record<string, number>)[key] ?? 0
                    return (
                      <div key={key} className="border border-gray-100 rounded-lg p-2 text-center">
                        <p className="text-[10px] text-gray-400">{label}</p>
                        <span className={`text-sm font-bold font-mono px-1.5 py-0.5 rounded ${scoreColor(val)}`}>
                          {(val * 100).toFixed(1)}%
                        </span>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}

            {/* 响应指标卡片 */}
            {result.response && (
              <div className="space-y-2">
                <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider flex items-center gap-1">
                  <FileBarChart size={12} /> 响应指标
                </p>
                <div className="grid grid-cols-4 gap-2">
                  {[
                    { key: 'rouge_l', label: 'ROUGE-L', invert: false },
                    { key: 'bleu', label: 'BLEU', invert: false },
                    { key: 'hallucination_rate', label: '幻觉率', invert: true },
                    { key: 'completeness', label: '完整性', invert: false },
                  ].map(({ key, label, invert }) => {
                    const val = (result.response as unknown as Record<string, number | undefined>)[key]
                    if (val === undefined || val === null) return <div key={key} />
                    const displayVal = invert ? 1 - val : val
                    return (
                      <div key={key} className="border border-gray-100 rounded-lg p-2 text-center">
                        <p className="text-[10px] text-gray-400">{label}</p>
                        <span className={`text-sm font-bold font-mono px-1.5 py-0.5 rounded ${scoreColor(displayVal)}`}>
                          {(val * 100).toFixed(1)}%
                        </span>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}

            {/* RAGAS原始指标（保留兼容） */}
            {result.scores && Object.keys(result.scores).length > 0 && (
              <div className="space-y-2">
                <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider">RAGAS 原始指标</p>
                {Object.entries(result.scores).map(([key, val]) => (
                  <div key={key}>
                    <p className="text-xs text-gray-500 mb-1">{metricLabel(key)}</p>
                    {scoreBar(val)}
                  </div>
                ))}
              </div>
            )}

            {/* 分类型指标 */}
            {result.type_scores && Object.keys(result.type_scores).length > 0 && (
              <div className="space-y-3">
                <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider">分类型指标</p>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  {Object.entries(result.type_scores).map(([qtype, scores]) => {
                    const Icon = TYPE_ICONS[qtype] || Search
                    return (
                      <div key={qtype} className="border border-gray-100 rounded-lg p-3">
                        <div className="flex items-center gap-1.5 mb-2">
                          <Icon size={14} className="text-gray-500" />
                          <span className="text-xs font-semibold text-gray-700">{TYPE_LABELS[qtype] || qtype}</span>
                        </div>
                        {Object.entries(scores).map(([key, val]) => (
                          <div key={key} className="flex items-center justify-between text-xs mb-1">
                            <span className="text-gray-500">{metricLabel(key)}</span>
                            <span className={`px-1.5 py-0.5 rounded font-mono ${scoreColor(val)}`}>{val.toFixed(4)}</span>
                          </div>
                        ))}
                      </div>
                    )
                  })}
                </div>
              </div>
            )}

            {/* 每题详情 */}
            {result.details && result.details.length > 0 && (
              <div className="space-y-2">
                <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider">逐题详情</p>
                {result.details.map((d, i) => {
                  const isExpanded = expandedDetail === i
                  const TypeIcon = TYPE_ICONS[d.question_type] || Search
                  return (
                    <div key={i} className="border border-gray-100 rounded-lg overflow-hidden">
                      <button
                        onClick={() => setExpandedDetail(isExpanded ? null : i)}
                        className="w-full flex items-center gap-2 px-3 py-2 text-left hover:bg-gray-50 transition"
                      >
                        <TypeIcon size={14} className="text-gray-400 flex-shrink-0" />
                        <span className="text-sm text-gray-700 flex-1 truncate">{d.question}</span>
                        <span className={`text-xs px-1.5 py-0.5 rounded ${scoreColor(d.source_count > 0 ? 0.8 : 0.2)}`}>
                          {d.source_count}条来源
                        </span>
                        <span className="text-xs text-gray-400">{TYPE_LABELS[d.question_type] || d.question_type}</span>
                        {isExpanded ? <ChevronUp size={14} className="text-gray-400" /> : <ChevronDown size={14} className="text-gray-400" />}
                      </button>
                      {isExpanded && (
                        <div className="px-3 pb-3 space-y-2 border-t border-gray-100">
                          {/* 检索/响应指标 */}
                          <div className="grid grid-cols-2 gap-2 pt-2">
                            {d.retrieval && (
                              <div className="bg-gray-50 rounded p-2">
                                <p className="text-[10px] font-semibold text-gray-500 mb-1">检索指标</p>
                                {Object.entries(d.retrieval).map(([k, v]) => (
                                  <div key={k} className="flex justify-between text-[10px]">
                                    <span className="text-gray-500">{k}</span>
                                    <span className="font-mono">{typeof v === 'number' ? v.toFixed(4) : v}</span>
                                  </div>
                                ))}
                              </div>
                            )}
                            {d.response && (
                              <div className="bg-gray-50 rounded p-2">
                                <p className="text-[10px] font-semibold text-gray-500 mb-1">响应指标</p>
                                {Object.entries(d.response).map(([k, v]) => (
                                  <div key={k} className="flex justify-between text-[10px]">
                                    <span className="text-gray-500">{k}</span>
                                    <span className="font-mono">{typeof v === 'number' ? v.toFixed(4) : v}</span>
                                  </div>
                                ))}
                              </div>
                            )}
                          </div>
                          <div>
                            <p className="text-xs font-semibold text-gray-500 mb-1">系统回答</p>
                            <p className="text-xs text-gray-700 bg-gray-50 rounded p-2 whitespace-pre-wrap">{d.answer}</p>
                          </div>
                          <div>
                            <p className="text-xs font-semibold text-gray-500 mb-1">标准答案</p>
                            <p className="text-xs text-gray-600 bg-blue-50 rounded p-2 whitespace-pre-wrap">{d.ground_truth}</p>
                          </div>
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
            )}
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
            <p className="font-medium mb-1">对比实验方法</p>
            <p>在后端运行 <code className="bg-blue-100 px-1 rounded">python run_v3_comparison.py</code> 可自动执行 V1/V2/V3/V4 四组对比实验。</p>
            <p className="mt-1">也可指定 <code className="bg-blue-100 px-1 rounded">--only agent</code> 只运行V4，<code className="bg-blue-100 px-1 rounded">--sample 5</code> 限制样本数。</p>
          </div>
        </div>

        {comparison && comparison.results && Object.keys(comparison.results).length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b-2 border-gray-200">
                  <th className="text-left py-3 px-3 text-gray-600 font-semibold">策略</th>
                  {comparison.metrics.map((m) => (
                    <th key={m} className="text-center py-3 px-3 text-gray-600 font-semibold">{metricLabel(m)}</th>
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

      {/* 可观测性面板 */}
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-semibold text-gray-700 flex items-center gap-2">
            <Eye size={18} className="text-purple-600" />
            可观测性面板
          </h3>
          <button
            onClick={() => getObservability().then(setObservability).catch(() => {})}
            className="flex items-center gap-2 rounded-lg bg-purple-600 px-4 py-2 text-sm text-white hover:bg-purple-700 transition"
          >
            <Eye size={16} />
            查看概览
          </button>
        </div>

        {observability && !observability.message ? (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* 检索质量 */}
            <div className="border border-blue-100 rounded-lg p-4 bg-blue-50/50">
              <div className="flex items-center gap-1.5 mb-3">
                <Search size={16} className="text-blue-600" />
                <p className="text-sm font-semibold text-blue-800">检索质量</p>
              </div>
              {[
                { key: 'recall@5', label: '召回率 R@5' },
                { key: 'precision@5', label: '精确率 P@5' },
                { key: 'corpus_coverage', label: '语料覆盖率' },
              ].map(({ key, label }) => {
                const val = (observability.retrieval_quality as unknown as Record<string, number>)[key] ?? 0
                return (
                  <div key={key} className="flex items-center justify-between mb-1">
                    <span className="text-xs text-gray-600">{label}</span>
                    <span className={`text-xs font-mono font-semibold px-1.5 py-0.5 rounded ${scoreColor(val)}`}>
                      {(val * 100).toFixed(1)}%
                    </span>
                  </div>
                )
              })}
            </div>

            {/* 生成质量 */}
            <div className="border border-green-100 rounded-lg p-4 bg-green-50/50">
              <div className="flex items-center gap-1.5 mb-3">
                <Shield size={16} className="text-green-600" />
                <p className="text-sm font-semibold text-green-800">生成质量</p>
              </div>
              {[
                { key: 'faithfulness', label: '忠实度' },
                { key: 'hallucination_rate', label: '幻觉率', invert: true },
                { key: 'completeness', label: '完整性' },
              ].map(({ key, label, invert }) => {
                const val = (observability.generation_quality as unknown as Record<string, number>)[key] ?? 0
                const displayVal = invert ? 1 - val : val
                return (
                  <div key={key} className="flex items-center justify-between mb-1">
                    <span className="text-xs text-gray-600">{label}</span>
                    <span className={`text-xs font-mono font-semibold px-1.5 py-0.5 rounded ${scoreColor(displayVal)}`}>
                      {(val * 100).toFixed(1)}%
                    </span>
                  </div>
                )
              })}
            </div>

            {/* 业务指标 */}
            <div className="border border-orange-100 rounded-lg p-4 bg-orange-50/50">
              <div className="flex items-center gap-1.5 mb-3">
                <Activity size={16} className="text-orange-600" />
                <p className="text-sm font-semibold text-orange-800">业务指标</p>
              </div>
              {[
                { key: 'resolution_rate', label: '问题解决率' },
                { key: 'first_answer_usability', label: '首次回答可用率' },
              ].map(({ key, label }) => {
                const val = (observability.business as unknown as Record<string, number>)[key] ?? 0
                return (
                  <div key={key} className="flex items-center justify-between mb-1">
                    <span className="text-xs text-gray-600">{label}</span>
                    <span className={`text-xs font-mono font-semibold px-1.5 py-0.5 rounded ${scoreColor(val)}`}>
                      {(val * 100).toFixed(1)}%
                    </span>
                  </div>
                )
              })}
            </div>
          </div>
        ) : !observability ? (
          <div className="text-center text-gray-400 text-sm py-6">
            点击"查看概览"获取三维度状态
          </div>
        ) : (
          <div className="text-center text-gray-400 text-sm py-6">
            {observability.message}
          </div>
        )}
      </div>

      {/* 断点续评 */}
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-semibold text-gray-700 flex items-center gap-2">
            <ListChecks size={18} className="text-indigo-600" />
            断点续评
          </h3>
          <button
            onClick={() => listPersistentTasks().then(r => setPTasks(r.tasks)).catch(() => {})}
            className="flex items-center gap-2 rounded-lg bg-indigo-600 px-4 py-2 text-sm text-white hover:bg-indigo-700 transition"
          >
            <ListChecks size={16} />
            刷新任务列表
          </button>
        </div>

        <div className="flex items-start gap-2 bg-indigo-50 border border-indigo-100 rounded-lg p-3 mb-4">
          <Info size={16} className="text-indigo-500 flex-shrink-0 mt-0.5" />
          <div className="text-xs text-indigo-700">
            <p className="font-medium mb-1">断点续评说明</p>
            <p>支持崩溃后从断点继续评估，每题结果实时保存。选择「继续评估」跳过已完成题目，「重新评估」清空从头开始。</p>
          </div>
        </div>

        {/* 创建新任务 */}
        <div className="flex flex-wrap items-center gap-3 mb-4 p-3 bg-gray-50 rounded-lg">
          <span className="text-sm text-gray-600 font-medium">新建任务:</span>
          <select
            value={ragMode}
            onChange={(e) => setRagMode(e.target.value)}
            className="rounded border border-gray-300 px-2 py-1 text-sm"
          >
            <option value="">当前配置</option>
            <option value="simple">V1/V2 简单链路</option>
            <option value="crag">V3 CRAG纠错</option>
            <option value="agent">V4 Agentic RAG</option>
          </select>
          <select
            value={sampleCount}
            onChange={(e) => setSampleCount(Number(e.target.value))}
            className="rounded border border-gray-300 px-2 py-1 text-sm"
          >
            <option value={3}>3题</option>
            <option value={5}>5题</option>
            <option value={10}>10题</option>
            <option value={20}>全部(24)</option>
          </select>
          <button
            onClick={async () => {
              try {
                const { task_id } = await createPersistentTask(ragMode || undefined, sampleCount, questionType || undefined)
                await startPersistentTask(task_id)
                setPRunning(true)
                startPPolling(task_id)
                listPersistentTasks().then(r => setPTasks(r.tasks)).catch(() => {})
              } catch (err) { alert(`创建任务失败: ${(err as Error).message}`) }
            }}
            disabled={pRunning}
            className="flex items-center gap-1.5 rounded-lg bg-emerald-600 px-3 py-1.5 text-sm text-white hover:bg-emerald-700 disabled:opacity-50 transition"
          >
            <Play size={14} /> 创建并启动
          </button>
        </div>

        {/* 任务列表 */}
        {pTasks.length > 0 ? (
          <div className="space-y-2">
            {pTasks.map(task => {
              const statusLabel: Record<string, { text: string; color: string }> = {
                created: { text: '已创建', color: 'bg-gray-100 text-gray-600' },
                running: { text: '运行中', color: 'bg-blue-100 text-blue-600' },
                paused: { text: '已暂停', color: 'bg-yellow-100 text-yellow-600' },
                completed: { text: '已完成', color: 'bg-green-100 text-green-600' },
                failed: { text: '失败', color: 'bg-red-100 text-red-600' },
              }
              const st = statusLabel[task.status] || statusLabel.created
              const done = task.completed_indices.length
              const fail = task.failed_indices.length
              const total = task.total_questions || '?'
              const pct = typeof total === 'number' && total > 0 ? Math.round(done / total * 100) : 0
              return (
                <div key={task.task_id} className="border border-gray-100 rounded-lg p-3">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <span className={`text-xs px-2 py-0.5 rounded font-medium ${st.color}`}>{st.text}</span>
                      <span className="text-xs text-gray-500 font-mono">{task.task_id.slice(0, 20)}...</span>
                      {task.rag_mode && <span className="text-xs text-gray-400">{STRATEGY_LABELS[task.rag_mode] || task.rag_mode}</span>}
                    </div>
                    <div className="flex items-center gap-1.5">
                      {/* 继续评估 */}
                      {(task.status === 'paused' || task.status === 'failed') && done < (task.total_questions || 0) && (
                        <button
                          onClick={async () => {
                            try {
                              await resumePersistentTask(task.task_id)
                              setPRunning(true)
                              startPPolling(task.task_id)
                            } catch (err) { alert(`续评失败: ${(err as Error).message}`) }
                          }}
                          className="flex items-center gap-1 rounded bg-blue-500 px-2 py-1 text-xs text-white hover:bg-blue-600 transition"
                        >
                          <FastForward size={12} /> 继续
                        </button>
                      )}
                      {/* 重新评估 */}
                      {task.status !== 'running' && (
                        <button
                          onClick={async () => {
                            if (!confirm('重新评估将清空该任务所有缓存，确认？')) return
                            try {
                              await restartPersistentTask(task.task_id)
                              setPRunning(true)
                              startPPolling(task.task_id)
                            } catch (err) { alert(`重新评估失败: ${(err as Error).message}`) }
                          }}
                          className="flex items-center gap-1 rounded bg-orange-500 px-2 py-1 text-xs text-white hover:bg-orange-600 transition"
                        >
                          <RotateCcw size={12} /> 重评
                        </button>
                      )}
                      {/* 重试失败 */}
                      {task.status !== 'running' && fail > 0 && (
                        <button
                          onClick={async () => {
                            try {
                              await retryFailedPersistentTask(task.task_id)
                              setPRunning(true)
                              startPPolling(task.task_id)
                            } catch (err) { alert(`重试失败: ${(err as Error).message}`) }
                          }}
                          className="flex items-center gap-1 rounded bg-red-500 px-2 py-1 text-xs text-white hover:bg-red-600 transition"
                        >
                          <RefreshCw size={12} /> 重试({fail})
                        </button>
                      )}
                      {/* 查看报告 */}
                      {task.status === 'completed' && (
                        <button
                          onClick={() => getPersistentReport(task.task_id).then(setPReport).catch(() => {})}
                          className="flex items-center gap-1 rounded bg-green-500 px-2 py-1 text-xs text-white hover:bg-green-600 transition"
                        >
                          <FileBarChart size={12} /> 报告
                        </button>
                      )}
                      {/* 查看进度 */}
                      {task.status === 'running' && (
                        <button
                          onClick={() => getPersistentProgress(task.task_id).then(setPProgress).catch(() => {})}
                          className="flex items-center gap-1 rounded bg-gray-500 px-2 py-1 text-xs text-white hover:bg-gray-600 transition"
                        >
                          进度
                        </button>
                      )}
                    </div>
                  </div>
                  {/* 进度条 */}
                  {typeof total === 'number' && total > 0 && (
                    <div>
                      <div className="flex items-center justify-between text-xs text-gray-500 mb-1">
                        <span>{done}/{total} 完成{fail > 0 ? `, ${fail} 失败` : ''}</span>
                        <span>{pct}%</span>
                      </div>
                      <div className="h-1.5 bg-gray-200 rounded-full overflow-hidden">
                        <div className="h-full bg-blue-500 rounded-full transition-all" style={{ width: `${pct}%` }} />
                      </div>
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        ) : (
          <div className="text-center text-gray-400 text-sm py-4">
            暂无任务，点击"刷新任务列表"或创建新任务
          </div>
        )}

        {/* 实时进度 */}
        {pProgress && (
          <div className="mt-4 p-3 bg-blue-50 rounded-lg">
            <p className="text-sm font-semibold text-blue-800 mb-2">
              任务进度: {pProgress.task_id.slice(0, 20)}...
            </p>
            <div className="grid grid-cols-4 gap-2 text-center">
              <div><p className="text-lg font-bold text-blue-700">{pProgress.total}</p><p className="text-[10px] text-gray-500">总题数</p></div>
              <div><p className="text-lg font-bold text-green-600">{pProgress.completed}</p><p className="text-[10px] text-gray-500">已完成</p></div>
              <div><p className="text-lg font-bold text-red-500">{pProgress.failed}</p><p className="text-[10px] text-gray-500">失败</p></div>
              <div><p className="text-lg font-bold text-orange-500">{pProgress.remaining}</p><p className="text-[10px] text-gray-500">剩余</p></div>
            </div>
            <div className="mt-2 h-2 bg-gray-200 rounded-full overflow-hidden">
              <div className="h-full bg-blue-500 rounded-full transition-all" style={{ width: `${pProgress.progress_pct}%` }} />
            </div>
            <p className="text-xs text-gray-500 mt-1 text-right">{pProgress.progress_pct}%</p>
          </div>
        )}

        {/* 续评报告 */}
        {pReport && (
          <div className="mt-4 p-4 bg-green-50 rounded-lg border border-green-100">
            <p className="text-sm font-semibold text-green-800 mb-3">评估报告</p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
              <div className="text-center"><p className="text-lg font-bold text-gray-700">{pReport.summary.total}</p><p className="text-[10px] text-gray-500">总题数</p></div>
              <div className="text-center"><p className="text-lg font-bold text-green-600">{pReport.summary.success}</p><p className="text-[10px] text-gray-500">成功</p></div>
              <div className="text-center"><p className="text-lg font-bold text-red-500">{pReport.summary.failed}</p><p className="text-[10px] text-gray-500">失败</p></div>
              <div className="text-center"><p className="text-lg font-bold text-blue-600">{pReport.summary.success_rate}%</p><p className="text-[10px] text-gray-500">成功率</p></div>
            </div>
            {pReport.retrieval_avg && (
              <div className="mb-2">
                <p className="text-xs font-semibold text-gray-500 mb-1">检索指标平均</p>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(pReport.retrieval_avg).map(([k, v]) => (
                    <span key={k} className={`text-xs font-mono px-2 py-0.5 rounded ${scoreColor(v)}`}>{k}: {(v * 100).toFixed(1)}%</span>
                  ))}
                </div>
              </div>
            )}
            {pReport.response_avg && (
              <div className="mb-2">
                <p className="text-xs font-semibold text-gray-500 mb-1">响应指标平均</p>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(pReport.response_avg).map(([k, v]) => (
                    <span key={k} className={`text-xs font-mono px-2 py-0.5 rounded ${scoreColor(k === 'hallucination_rate' ? 1 - v : v)}`}>{k}: {(v * 100).toFixed(1)}%</span>
                  ))}
                </div>
              </div>
            )}
            {pReport.ragas_full && (
              <div className="mt-3 pt-3 border-t border-green-200">
                <p className="text-xs font-semibold text-gray-500 mb-2">完整RAGAS报告</p>
                {pReport.ragas_full.triad && (
                  <div className="grid grid-cols-3 gap-2 mb-2">
                    {Object.entries(pReport.ragas_full.triad).map(([k, v]) => (
                      <div key={k} className="text-center border border-green-200 rounded p-2">
                        <p className="text-[10px] text-gray-500">{k}</p>
                        <span className={`text-sm font-bold font-mono px-1.5 py-0.5 rounded ${scoreColor(v as number)}`}>{((v as number) * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
            {pReport.failed_items && pReport.failed_items.length > 0 && (
              <div className="mt-2">
                <p className="text-xs font-semibold text-red-500 mb-1">失败条目</p>
                {pReport.failed_items.map((item, i) => (
                  <div key={i} className="text-xs text-red-600 bg-red-50 rounded p-1.5 mb-1">
                    #{item.index} {item.question} — {item.error}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Bad Case 追踪 */}
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-semibold text-gray-700 flex items-center gap-2">
            <AlertTriangle size={18} className="text-red-600" />
            Bad Case 追踪
          </h3>
          <button
            onClick={() => getBadCases(5).then(setBadCases).catch(() => {})}
            className="flex items-center gap-2 rounded-lg bg-red-600 px-4 py-2 text-sm text-white hover:bg-red-700 transition"
          >
            <AlertTriangle size={16} />
            查看Bad Case
          </button>
        </div>

        {badCases.length > 0 ? (
          <div className="space-y-2">
            {badCases.map((bc, i) => {
              const isExpanded = expandedBadCase === i
              const TypeIcon = TYPE_ICONS[bc.question_type] || Search
              return (
                <div key={i} className="border border-red-100 rounded-lg overflow-hidden">
                  <button
                    onClick={() => setExpandedBadCase(isExpanded ? null : i)}
                    className="w-full flex items-center gap-2 px-3 py-2 text-left hover:bg-red-50/50 transition"
                  >
                    <span className="text-xs font-bold text-red-500">#{i + 1}</span>
                    <TypeIcon size={14} className="text-gray-400 flex-shrink-0" />
                    <span className="text-sm text-gray-700 flex-1 truncate">{bc.question}</span>
                    <span className="text-xs text-gray-400">{TYPE_LABELS[bc.question_type] || bc.question_type}</span>
                    {isExpanded ? <ChevronUp size={14} className="text-gray-400" /> : <ChevronDown size={14} className="text-gray-400" />}
                  </button>
                  {isExpanded && (
                    <div className="px-3 pb-3 space-y-2 border-t border-red-100">
                      <div>
                        <p className="text-xs font-semibold text-red-500 mb-1">系统回答</p>
                        <p className="text-xs text-gray-700 bg-red-50/50 rounded p-2 whitespace-pre-wrap">{bc.answer}</p>
                      </div>
                      <div>
                        <p className="text-xs font-semibold text-blue-500 mb-1">标准答案</p>
                        <p className="text-xs text-gray-600 bg-blue-50 rounded p-2 whitespace-pre-wrap">{bc.ground_truth}</p>
                      </div>
                      {bc.response && (
                        <div className="bg-gray-50 rounded p-2">
                          <p className="text-[10px] font-semibold text-gray-500 mb-1">响应指标</p>
                          {Object.entries(bc.response).map(([k, v]) => (
                            <div key={k} className="flex justify-between text-[10px]">
                              <span className="text-gray-500">{k}</span>
                              <span className="font-mono">{typeof v === 'number' ? v.toFixed(4) : v}</span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        ) : (
          <div className="text-center text-gray-400 text-sm py-6">
            点击"查看Bad Case"获取得分最低的题目
          </div>
        )}
      </div>
    </div>
  )
}
