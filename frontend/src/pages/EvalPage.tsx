import { useState, useEffect, useRef } from 'react'
import { BarChart3, Play, FileBarChart, Info, ChevronDown, ChevronUp, Search, Calculator, GitCompare, Eye, AlertTriangle, Shield, Activity, RotateCcw, FastForward, RefreshCw, ListChecks, Square, Trash2, Gauge, Layers, ArrowLeftRight, Sparkles } from 'lucide-react'
import { getEvaluationReport, getObservability, getBadCases, createPersistentTask, startPersistentTask, resumePersistentTask, restartPersistentTask, retryFailedPersistentTask, forceStopPersistentTask, deletePersistentTask, listPersistentTasks, getPersistentProgress, getPersistentReport, type EvalComparison, type Observability, type BadCase, type PersistentTask, type PersistentProgress, type PersistentReport } from '../api'

const STRATEGY_LABELS: Record<string, string> = {
  vector: 'V1 纯向量',
  hybrid: 'V2 混合检索',
  reranked: 'V2 重排序',
  crag: 'V3 CRAG纠错',
  simple: 'V1/V2 简单链路',
  agent: 'V4 Agentic RAG',
}

const STRATEGY_COLORS: Record<string, string> = {
  vector: 'text-brand-600 bg-brand-50 border-brand-200',
  hybrid: 'text-accent-600 bg-accent-50 border-accent-200',
  reranked: 'text-cyan-600 bg-cyan-50 border-cyan-100',
  crag: 'text-amber-600 bg-amber-50 border-amber-200',
  simple: 'text-slate-600 bg-slate-50 border-slate-200',
  agent: 'text-emerald-600 bg-emerald-50 border-emerald-100',
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

const POLL_INTERVAL = 5000

export default function EvalPage() {
  const [comparison, setComparison] = useState<EvalComparison | null>(null)
  const [observability, setObservability] = useState<Observability | null>(null)
  const [badCases, setBadCases] = useState<BadCase[]>([])
  const [sampleCount, setSampleCount] = useState(5)
  const [ragMode, setRagMode] = useState<string>('')
  const [questionType, setQuestionType] = useState<string>('')
  const [expandedBadCase, setExpandedBadCase] = useState<number | null>(null)

  const [pTasks, setPTasks] = useState<PersistentTask[]>([])
  const [pProgress, setPProgress] = useState<PersistentProgress | null>(null)
  const [pReport, setPReport] = useState<PersistentReport | null>(null)
  const [pRunning, setPRunning] = useState(false)
  const pPollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const startPPolling = (taskId: string) => {
    if (pPollRef.current) clearInterval(pPollRef.current)
    pPollRef.current = setInterval(async () => {
      try {
        const prog = await getPersistentProgress(taskId)
        setPProgress(prog)
        if (prog.status !== 'running' && prog.status !== 'scoring') {
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

  useEffect(() => {
    listPersistentTasks().then(r => setPTasks(r.tasks)).catch(() => {})
    return () => { if (pPollRef.current) clearInterval(pPollRef.current) }
  }, [])

  const handleCompare = async () => {
    try {
      const report = await getEvaluationReport()
      setComparison(report)
    } catch (err) {
      alert(`获取报告失败: ${(err as Error).message}`)
    }
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
    <div className="max-w-5xl mx-auto p-5 lg:p-6 space-y-5 overflow-y-auto h-full pb-8">
      {/* ── 页面标题 ── */}
      <div className="flex items-center gap-3">
        <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-brand-500 to-accent-500 flex items-center justify-center shadow-sm shadow-brand-200/40">
          <BarChart3 size={20} className="text-white" />
        </div>
        <div>
          <h2 className="text-lg font-bold text-gradient-brand">评估系统</h2>
          <p className="text-xs text-slate-400 mt-0.5">多策略评估、对比实验与 Bad Case 追踪</p>
        </div>
      </div>

      {/* ── 评估任务 ── */}
      <div className="bg-white rounded-xl border border-slate-200/80 shadow-card overflow-hidden">
        <div className="px-5 py-4 border-b border-brand-100/50 flex items-center justify-between bg-brand-50/30">
          <h3 className="font-semibold text-slate-700 flex items-center gap-2 text-sm">
            <ListChecks size={16} className="text-accent-500" />
            评估任务
          </h3>
          <div className="flex items-center gap-2">
            <button
              onClick={() => {
                listPersistentTasks().then(r => setPTasks(r.tasks)).catch(() => {})
                getEvaluationReport().then(setComparison).catch(() => {})
                getObservability().then(setObservability).catch(() => {})
                getBadCases().then(setBadCases).catch(() => {})
              }}
              className="inline-flex items-center gap-1.5 rounded-lg btn-brand px-3.5 py-2 text-xs transition-all duration-200"
            >
              <RefreshCw size={14} />
              刷新
            </button>
          </div>
        </div>

        {/* 提示 */}
        <div className="mx-5 mt-4 flex items-start gap-2.5 rounded-xl bg-brand-50/80 border border-brand-200/60 p-3.5">
          <Info size={15} className="text-brand-500 flex-shrink-0 mt-0.5" />
          <div className="text-xs text-brand-700 leading-relaxed">
            <p className="font-semibold mb-0.5">评估说明</p>
            <p>每题结果实时保存，支持崩溃后从断点继续。选择「继续」跳过已完成题目，「重评」清空从头开始。</p>
          </div>
        </div>

        {/* 创建新任务 */}
        <div className="mx-5 mt-4 flex flex-wrap items-center gap-3 p-3.5 bg-slate-50/80 rounded-xl border border-slate-200/80">
          <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">新建任务</span>
          <select
            value={ragMode}
            onChange={(e) => setRagMode(e.target.value)}
            className="rounded-lg border border-slate-200 bg-white px-2.5 py-1.5 text-xs text-slate-700 focus:outline-none focus:ring-2 focus:ring-brand-400/30 focus:border-brand-400"
          >
            <option value="">当前配置</option>
            <option value="simple">V1/V2 简单链路</option>
            <option value="crag">V3 CRAG纠错</option>
            <option value="agent">V4 Agentic RAG</option>
          </select>
          <select
            value={sampleCount}
            onChange={(e) => setSampleCount(Number(e.target.value))}
            className="rounded-lg border border-slate-200 bg-white px-2.5 py-1.5 text-xs text-slate-700 focus:outline-none focus:ring-2 focus:ring-brand-400/30 focus:border-brand-400"
          >
            <option value={3}>3题</option>
            <option value={5}>5题</option>
            <option value={10}>10题</option>
            <option value={20}>全部(24)</option>
          </select>
          <select
            value={questionType}
            onChange={(e) => setQuestionType(e.target.value)}
            className="rounded-lg border border-slate-200 bg-white px-2.5 py-1.5 text-xs text-slate-700 focus:outline-none focus:ring-2 focus:ring-brand-400/30 focus:border-brand-400"
          >
            <option value="">全部类型</option>
            <option value="retrieve">法条查询</option>
            <option value="calculate">计算类</option>
            <option value="compare">对比类</option>
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
            className="inline-flex items-center gap-1.5 rounded-lg btn-brand px-3.5 py-1.5 text-xs disabled:opacity-50 transition-all duration-200"
          >
            <Play size={13} /> 创建并启动
          </button>
        </div>

        {/* 任务列表 */}
        <div className="px-5 pb-5 mt-3">
          {pTasks.length > 0 ? (
            <div className="space-y-2.5">
              {pTasks.map(task => {
                const statusLabel: Record<string, { text: string; color: string }> = {
                  created: { text: '已创建', color: 'bg-slate-100 text-slate-600 border-slate-200' },
                  running: { text: '运行中', color: 'bg-brand-100 text-brand-600 border-brand-200' },
                  scoring: { text: '评分中', color: 'bg-accent-100 text-accent-600 border-accent-200' },
                  paused: { text: '已暂停', color: 'bg-yellow-100 text-yellow-600 border-yellow-200' },
                  completed: { text: '已完成', color: 'bg-emerald-100 text-emerald-600 border-emerald-100' },
                  failed: { text: '失败', color: 'bg-red-100 text-red-600 border-red-200' },
                }
                const st = statusLabel[task.status] || statusLabel.created
                const done = task.completed_indices.length
                const fail = task.failed_indices.length
                const total = task.total_questions || '?'
                const pct = typeof total === 'number' && total > 0 ? Math.round(done / total * 100) : 0
                return (
                  <div key={task.task_id} className="border border-slate-100/80 rounded-xl p-4 bg-white hover:border-brand-200 transition-colors shadow-sm">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-2.5 flex-wrap">
                        <span className={`text-[10px] font-semibold px-2 py-0.5 rounded-full border ${st.color}`}>{st.text}</span>
                        <span className="text-[11px] text-slate-400 font-mono">{task.task_id.slice(0, 12)}...</span>
                        {task.rag_mode && (
                          <span className={`text-[10px] font-medium px-1.5 py-0.5 rounded-full border ${STRATEGY_COLORS[task.rag_mode] || 'text-gray-500 bg-gray-50 border-gray-200'}`}>
                            {STRATEGY_LABELS[task.rag_mode] || task.rag_mode}
                          </span>
                        )}
                      </div>
                      <div className="flex items-center gap-1.5 flex-wrap justify-end">
                        {(task.status === 'paused' || task.status === 'failed') && (
                          <button
                            disabled={pRunning}
                            onClick={async () => {
                              try {
                                setPRunning(true)
                                await resumePersistentTask(task.task_id)
                                startPPolling(task.task_id)
                              } catch (err) { setPRunning(false); alert(`续评失败: ${(err as Error).message}`) }
                            }}
                            className={`inline-flex items-center gap-1 rounded-lg px-2.5 py-1.5 text-[11px] font-medium text-white transition-all duration-200 ${pRunning ? 'bg-brand-300 cursor-not-allowed' : 'bg-brand-500 hover:bg-brand-600 shadow-sm'}`}
                          >
                            <FastForward size={11} /> 继续
                          </button>
                        )}
                        {task.status !== 'running' && task.status !== 'scoring' && (
                          <button
                            onClick={async () => {
                              if (!confirm('重新评估将清空该任务所有缓存，确认？')) return
                              try {
                                await restartPersistentTask(task.task_id)
                                setPRunning(true)
                                startPPolling(task.task_id)
                              } catch (err) { alert(`重新评估失败: ${(err as Error).message}`) }
                            }}
                            className="inline-flex items-center gap-1 rounded-lg bg-amber-500 px-2.5 py-1.5 text-[11px] font-medium text-white hover:bg-amber-600 transition-all duration-200 shadow-sm"
                          >
                            <RotateCcw size={11} /> 重评
                          </button>
                        )}
                        {task.status !== 'running' && task.status !== 'scoring' && fail > 0 && (
                          <button
                            onClick={async () => {
                              try {
                                await retryFailedPersistentTask(task.task_id)
                                setPRunning(true)
                                startPPolling(task.task_id)
                              } catch (err) { alert(`重试失败: ${(err as Error).message}`) }
                            }}
                            className="inline-flex items-center gap-1 rounded-lg bg-red-500 px-2.5 py-1.5 text-[11px] font-medium text-white hover:bg-red-600 transition-all duration-200 shadow-sm"
                          >
                            <RefreshCw size={11} /> 重试({fail})
                          </button>
                        )}
                        {task.status === 'completed' && (
                          <button
                            onClick={() => getPersistentReport(task.task_id).then(setPReport).catch(() => {})}
                            className="inline-flex items-center gap-1 rounded-lg bg-green-500 px-2.5 py-1.5 text-[11px] font-medium text-white hover:bg-green-600 transition-all duration-200 shadow-sm"
                          >
                            <FileBarChart size={11} /> 报告
                          </button>
                        )}
                        {(task.status === 'running' || task.status === 'scoring') && (
                          <>
                            <button
                              onClick={() => getPersistentProgress(task.task_id).then(setPProgress).catch(() => {})}
                              className="inline-flex items-center gap-1 rounded-lg bg-slate-500 px-2.5 py-1.5 text-[11px] font-medium text-white hover:bg-slate-600 transition-all duration-200 shadow-sm"
                            >
                              进度
                            </button>
                            <button
                              onClick={async () => {
                                try {
                                  await forceStopPersistentTask(task.task_id)
                                  setPRunning(false)
                                  if (pPollRef.current) { clearInterval(pPollRef.current); pPollRef.current = null }
                                  listPersistentTasks().then(r => setPTasks(r.tasks)).catch(() => {})
                                } catch (err) { alert(`停止失败: ${(err as Error).message}`) }
                              }}
                              className="inline-flex items-center gap-1 rounded-lg bg-yellow-500 px-2.5 py-1.5 text-[11px] font-medium text-white hover:bg-yellow-600 transition-all duration-200 shadow-sm"
                            >
                              <Square size={11} /> 停止
                            </button>
                          </>
                        )}
                        {task.status !== 'running' && task.status !== 'scoring' && (
                          <button
                            onClick={async () => {
                              if (!confirm('确定删除此任务？')) return
                              try {
                                await deletePersistentTask(task.task_id)
                                listPersistentTasks().then(r => setPTasks(r.tasks)).catch(() => {})
                              } catch (err) { alert(`删除失败: ${(err as Error).message}`) }
                            }}
                            className="inline-flex items-center gap-1 rounded-lg px-2.5 py-1.5 text-[11px] font-medium text-slate-500 hover:bg-red-50 hover:text-red-500 border border-slate-200 hover:border-red-200 transition-all duration-200"
                          >
                            <Trash2 size={11} /> 删除
                          </button>
                        )}
                      </div>
                    </div>
                    {typeof total === 'number' && total > 0 && (
                      <div>
                        <div className="flex items-center justify-between text-[11px] text-slate-500 mb-1.5">
                          <span>
                            <span className="font-semibold text-slate-700">{done}</span>/{total} 完成
                            {fail > 0 && <span className="text-red-400 ml-1">· {fail} 失败</span>}
                          </span>
                          <span className="font-semibold">{pct}%</span>
                        </div>
                        <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                          <div
                            className="h-full rounded-full transition-all duration-500 ease-out"
                            style={{
                              width: `${pct}%`,
                              background: pct === 100
                                ? 'linear-gradient(90deg, #10b981, #059669)'
                                : 'linear-gradient(90deg, #3b82f6, #6366f1)'
                            }}
                          />
                        </div>
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-10 text-slate-400">
              <Sparkles size={28} className="text-brand-200 mb-2" />
              <p className="text-sm">暂无任务</p>
              <p className="text-xs text-slate-300 mt-1">点击上方按钮创建新的评估任务</p>
            </div>
          )}
        </div>

        {/* 实时进度 */}
        {pProgress && (
          <div className="mx-5 mb-5 p-4 bg-gradient-to-br from-brand-50 to-accent-50/30 rounded-xl border border-brand-200/60 animate-fade-in">
            <p className="text-sm font-semibold text-brand-800 mb-3 flex items-center gap-2">
              <Activity size={14} />
              任务进度: <span className="font-mono text-xs">{pProgress.task_id.slice(0, 12)}...</span>
              {pProgress.status === 'scoring' && (
                <span className="ml-auto text-accent-600 text-xs animate-pulse bg-accent-100 px-2 py-0.5 rounded-full">
                  RAGAS 评分中...
                </span>
              )}
            </p>
            <div className="grid grid-cols-4 gap-3 mb-3">
              {[
                { label: '总题数', value: pProgress.total, color: 'text-brand-700' },
                { label: '已完成', value: pProgress.completed, color: 'text-emerald-600' },
                { label: '失败', value: pProgress.failed, color: 'text-red-500' },
                { label: '剩余', value: pProgress.remaining, color: 'text-amber-500' },
              ].map((item) => (
                <div key={item.label} className="text-center bg-white/60 rounded-lg p-2.5 border border-brand-100/50">
                  <p className={`text-lg font-bold ${item.color}`}>{item.value}</p>
                  <p className="text-[10px] text-gray-500">{item.label}</p>
                </div>
              ))}
            </div>
            <div className="h-2.5 bg-slate-200/70 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-brand-500 to-accent-500 rounded-full transition-all duration-500 ease-out"
                style={{ width: `${pProgress.progress_pct}%` }}
              />
            </div>
            <p className="text-[11px] text-slate-500 mt-1.5 text-right font-medium">{pProgress.progress_pct}%</p>
          </div>
        )}

        {/* 评估报告 */}
        {pReport && (
          <div className="mx-5 mb-5 p-4 bg-gradient-to-br from-emerald-50 to-brand-50/20 rounded-xl border border-emerald-100/80 animate-fade-in">
            <p className="text-sm font-semibold text-emerald-800 mb-3 flex items-center gap-2">
              <FileBarChart size={14} />
              评估报告
            </p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
              {[
                { label: '总题数', value: pReport.summary.total, color: 'text-slate-700' },
                { label: '成功', value: pReport.summary.success, color: 'text-emerald-600' },
                { label: '失败', value: pReport.summary.failed, color: 'text-red-500' },
                { label: '成功率', value: `${pReport.summary.success_rate}%`, color: 'text-brand-600' },
              ].map((item) => (
                <div key={item.label} className="text-center bg-white/60 rounded-lg p-2.5 border border-emerald-100/50">
                  <p className={`text-lg font-bold ${item.color}`}>{item.value}</p>
                  <p className="text-[10px] text-gray-500">{item.label}</p>
                </div>
              ))}
            </div>
            {pReport.retrieval_avg && (
              <div className="mb-2.5 bg-white/60 rounded-lg p-3 border border-emerald-100/50">
                <p className="text-[11px] font-semibold text-slate-500 mb-2">检索指标平均</p>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(pReport.retrieval_avg).map(([k, v]) => (
                    <ScoreBadge key={k} label={k} value={v as number} />
                  ))}
                </div>
              </div>
            )}
            {pReport.response_avg && (
              <div className="mb-2.5 bg-white/60 rounded-lg p-3 border border-emerald-100/50">
                <p className="text-[11px] font-semibold text-slate-500 mb-2">响应指标平均</p>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(pReport.response_avg).map(([k, v]) => (
                    <ScoreBadge key={k} label={k} value={k === 'hallucination_rate' ? 1 - (v as number) : v as number} />
                  ))}
                </div>
              </div>
            )}
            {pReport.ragas_full && pReport.ragas_full.triad && (
              <div className="bg-white/60 rounded-lg p-3 border border-emerald-100/50">
                <p className="text-[11px] font-semibold text-slate-500 mb-2">完整 RAGAS 报告</p>
                <div className="grid grid-cols-3 gap-2">
                  {Object.entries(pReport.ragas_full.triad).map(([k, v]) => (
                    <div key={k} className="text-center bg-white rounded-lg p-2.5 border border-emerald-100/60 shadow-sm">
                      <p className="text-[10px] text-slate-500 mb-1">{k}</p>
                      <ScoreValue value={v as number} />
                    </div>
                  ))}
                </div>
              </div>
            )}
            {pReport.failed_items && pReport.failed_items.length > 0 && (
              <div className="mt-3">
                <p className="text-[11px] font-semibold text-red-500 mb-1.5">失败条目</p>
                {pReport.failed_items.map((item, i) => (
                  <div key={i} className="text-xs text-red-600 bg-red-50/80 rounded-lg p-2.5 mb-1.5 border border-red-100/60 leading-relaxed">
                    <span className="font-mono font-semibold">#{item.index}</span> {item.question}
                    <p className="text-red-400 text-[10px] mt-0.5">{item.error}</p>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* ── 对比报告 ── */}
      <div className="bg-white rounded-xl border border-slate-200/80 shadow-card overflow-hidden">
        <div className="px-5 py-4 border-b border-brand-100/50 flex items-center justify-between bg-brand-50/30">
          <h3 className="font-semibold text-slate-700 flex items-center gap-2 text-sm">
            <FileBarChart size={16} className="text-brand-500" />
            策略对比报告
          </h3>
          <button
            onClick={handleCompare}
            className="inline-flex items-center gap-1.5 rounded-lg btn-brand px-3.5 py-2 text-xs transition-all duration-200"
          >
            <FileBarChart size={14} />
            查看对比
          </button>
        </div>

        <div className="px-5 pt-4">
          <div className="flex items-start gap-2.5 rounded-xl bg-brand-50/80 border border-brand-200/60 p-3.5 mb-4">
            <Info size={15} className="text-brand-500 flex-shrink-0 mt-0.5" />
            <div className="text-xs text-brand-700 leading-relaxed">
              <p className="font-semibold mb-0.5">对比实验方法</p>
              <p>在后端运行 <code className="bg-brand-100/80 px-1.5 py-0.5 rounded text-[10px] font-mono">python run_v3_comparison.py</code> 可自动执行 V1/V2/V3/V4 四组对比实验。</p>
              <p className="mt-0.5">也可指定 <code className="bg-brand-100/80 px-1.5 py-0.5 rounded text-[10px] font-mono">--only agent</code> 只运行 V4，<code className="bg-brand-100/80 px-1.5 py-0.5 rounded text-[10px] font-mono">--sample 5</code> 限制样本数。</p>
            </div>
          </div>
        </div>

        <div className="px-5 pb-5">
          {comparison && comparison.results && Object.keys(comparison.results).length > 0 ? (
            <div className="overflow-x-auto rounded-xl border border-slate-100">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-brand-50/50 border-b-2 border-brand-100">
                    <th className="text-left py-3 px-4 text-xs font-semibold text-slate-500 uppercase tracking-wider">策略</th>
                    {comparison.metrics.map((m) => (
                      <th key={m} className="text-center py-3 px-4 text-xs font-semibold text-slate-500 uppercase tracking-wider">{metricLabel(m)}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(comparison.results).map(([type, scores], idx) => (
                    <tr key={type} className={`${idx % 2 === 0 ? 'bg-white' : 'bg-slate-50/50'} border-b border-slate-100/80 hover:bg-brand-50/30 transition-colors`}>
                      <td className="py-3.5 px-4">
                        <span className={`inline-flex items-center gap-1.5 text-xs font-semibold px-2.5 py-1 rounded-full border ${STRATEGY_COLORS[type] || 'text-gray-600 bg-gray-50 border-gray-200'}`}>
                          {type === 'reranked' || type === 'hybrid' ? <Gauge size={12} /> : type === 'vector' ? <Layers size={12} /> : <ArrowLeftRight size={12} />}
                          {STRATEGY_LABELS[type] || type}
                        </span>
                      </td>
                      {comparison.metrics.map((m) => {
                        const val = scores[m] || 0
                        return (
                          <td key={m} className="py-3.5 px-4 text-center">
                            <div className="flex flex-col items-center gap-1">
                              <ScoreValue value={val} />
                              <div className="w-16 h-1 bg-slate-100 rounded-full overflow-hidden">
                                <div
                                  className="h-full rounded-full transition-all duration-500"
                                  style={{
                                    width: `${val * 100}%`,
                                    background: val >= 0.8
                                      ? 'linear-gradient(90deg, #10b981, #059669)'
                                      : val >= 0.6
                                        ? 'linear-gradient(90deg, #f59e0b, #d97706)'
                                        : val >= 0.4
                                          ? 'linear-gradient(90deg, #f97316, #ea580c)'
                                          : 'linear-gradient(90deg, #ef4444, #dc2626)'
                                  }}
                                />
                              </div>
                            </div>
                          </td>
                        )
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-10 text-slate-400">
              <BarChart3 size={28} className="text-brand-200 mb-2" />
              <p className="text-sm">暂无对比数据</p>
              <p className="text-xs text-slate-300 mt-1">请先运行不同策略的评估</p>
            </div>
          )}

          {comparison?.message && (
            <p className="text-xs text-slate-400 text-center mt-3">{comparison.message}</p>
          )}
        </div>
      </div>

      {/* ── 可观测性面板 ── */}
      <div className="bg-white rounded-xl border border-slate-200/80 shadow-card overflow-hidden">
        <div className="px-5 py-4 border-b border-accent-100/50 flex items-center justify-between bg-accent-50/30">
          <h3 className="font-semibold text-slate-700 flex items-center gap-2 text-sm">
            <Eye size={16} className="text-accent-500" />
            可观测性面板
          </h3>
          <button
            onClick={() => getObservability().then(setObservability).catch(() => {})}
            className="inline-flex items-center gap-1.5 rounded-lg bg-accent-500 px-3.5 py-2 text-xs font-medium text-white hover:bg-accent-600 transition-all duration-200 shadow-sm"
          >
            <Eye size={14} />
            查看概览
          </button>
        </div>

        <div className="px-5 pb-5 pt-4">
          {observability && !observability.message ? (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* 检索质量 */}
              <div className="rounded-xl border border-brand-100/80 bg-gradient-to-b from-brand-50/60 to-white p-4">
                <div className="flex items-center gap-2 mb-3 pb-2 border-b border-brand-100/50">
                  <div className="w-7 h-7 rounded-lg bg-brand-100 flex items-center justify-center">
                    <Search size={14} className="text-brand-600" />
                  </div>
                  <p className="text-sm font-semibold text-brand-800">检索质量</p>
                </div>
                <div className="space-y-2.5">
                  {[
                    { key: 'recall@5', label: '召回率 R@5' },
                    { key: 'precision@5', label: '精确率 P@5' },
                    { key: 'corpus_coverage', label: '语料覆盖率' },
                  ].map(({ key, label }) => {
                    const val = (observability.retrieval_quality as unknown as Record<string, number>)[key] ?? 0
                    return <MetricRow key={key} label={label} value={val} />
                  })}
                </div>
              </div>

              {/* 生成质量 */}
              <div className="rounded-xl border border-emerald-100/80 bg-gradient-to-b from-emerald-50/60 to-white p-4">
                <div className="flex items-center gap-2 mb-3 pb-2 border-b border-emerald-100/50">
                  <div className="w-7 h-7 rounded-lg bg-emerald-100 flex items-center justify-center">
                    <Shield size={14} className="text-emerald-600" />
                  </div>
                  <p className="text-sm font-semibold text-emerald-800">生成质量</p>
                </div>
                <div className="space-y-2.5">
                  {[
                    { key: 'faithfulness', label: '忠实度' },
                    { key: 'hallucination_rate', label: '幻觉率', invert: true },
                    { key: 'completeness', label: '完整性' },
                  ].map(({ key, label, invert }) => {
                    const val = (observability.generation_quality as unknown as Record<string, number>)[key] ?? 0
                    return <MetricRow key={key} label={label} value={invert ? 1 - val : val} />
                  })}
                </div>
              </div>

              {/* 业务指标 */}
              <div className="rounded-xl border border-cyan-100/80 bg-gradient-to-b from-cyan-50/60 to-white p-4">
                <div className="flex items-center gap-2 mb-3 pb-2 border-b border-cyan-100/50">
                  <div className="w-7 h-7 rounded-lg bg-cyan-100 flex items-center justify-center">
                    <Activity size={14} className="text-cyan-600" />
                  </div>
                  <p className="text-sm font-semibold text-cyan-800">业务指标</p>
                </div>
                <div className="space-y-2.5">
                  {[
                    { key: 'resolution_rate', label: '问题解决率' },
                    { key: 'first_answer_usability', label: '首次回答可用率' },
                  ].map(({ key, label }) => {
                    const val = (observability.business as unknown as Record<string, number>)[key] ?? 0
                    return <MetricRow key={key} label={label} value={val} />
                  })}
                </div>
              </div>
            </div>
          ) : !observability ? (
            <div className="flex flex-col items-center justify-center py-8 text-slate-400">
              <Eye size={28} className="text-brand-200 mb-2" />
              <p className="text-sm">点击「查看概览」获取三维度状态</p>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-8 text-slate-400">
              <p className="text-sm">{observability.message}</p>
            </div>
          )}
        </div>
      </div>

      {/* ── Bad Case 追踪 ── */}
      <div className="bg-white rounded-xl border border-slate-200/80 shadow-card overflow-hidden">
        <div className="px-5 py-4 border-b border-red-100/50 flex items-center justify-between bg-red-50/20">
          <h3 className="font-semibold text-slate-700 flex items-center gap-2 text-sm">
            <AlertTriangle size={16} className="text-red-500" />
            Bad Case 追踪
          </h3>
          <button
            onClick={() => getBadCases(5).then(setBadCases).catch(() => {})}
            className="inline-flex items-center gap-1.5 rounded-lg bg-red-500 px-3.5 py-2 text-xs font-medium text-white hover:bg-red-600 transition-all duration-200 shadow-sm"
          >
            <AlertTriangle size={14} />
            查看 Bad Case
          </button>
        </div>

        <div className="px-5 pb-5 pt-4">
          {badCases.length > 0 ? (
            <div className="space-y-2.5">
              {badCases.map((bc, i) => {
                const isExpanded = expandedBadCase === i
                const TypeIcon = TYPE_ICONS[bc.question_type] || Search
                return (
                  <div key={i} className="border border-red-100/80 rounded-xl overflow-hidden bg-white shadow-sm hover:shadow-card-hover transition-shadow animate-scale-in">
                    <button
                      onClick={() => setExpandedBadCase(isExpanded ? null : i)}
                      className="w-full flex items-center gap-3 px-4 py-3.5 text-left hover:bg-red-50/40 transition-colors"
                    >
                      <span className="flex-shrink-0 w-6 h-6 rounded-lg bg-red-50 flex items-center justify-center text-[11px] font-bold text-red-500">#{i + 1}</span>
                      <TypeIcon size={15} className="text-slate-400 flex-shrink-0" />
                      <span className="text-sm text-slate-700 flex-1 truncate font-medium">{bc.question}</span>
                      <span className="text-[10px] text-slate-500 bg-slate-50 px-2 py-0.5 rounded-full font-medium border border-slate-200/60">
                        {TYPE_LABELS[bc.question_type] || bc.question_type}
                      </span>
                      {isExpanded ? <ChevronUp size={14} className="text-slate-400" /> : <ChevronDown size={14} className="text-slate-400" />}
                    </button>
                    {isExpanded && (
                      <div className="px-4 pb-4 space-y-3 border-t border-red-100/60 pt-3">
                        <div>
                          <p className="text-[11px] font-semibold text-red-500 mb-1.5 flex items-center gap-1">
                            <span className="w-1.5 h-1.5 rounded-full bg-red-400" />
                            系统回答
                          </p>
                          <div className="text-xs text-slate-700 bg-red-50/50 rounded-xl p-3 border border-red-100/50 whitespace-pre-wrap leading-relaxed">{bc.answer}</div>
                        </div>
                        <div>
                          <p className="text-[11px] font-semibold text-brand-500 mb-1.5 flex items-center gap-1">
                            <span className="w-1.5 h-1.5 rounded-full bg-brand-400" />
                            标准答案
                          </p>
                          <div className="text-xs text-slate-600 bg-brand-50/50 rounded-xl p-3 border border-brand-100/50 whitespace-pre-wrap leading-relaxed">{bc.ground_truth}</div>
                        </div>
                        {bc.response && (
                          <div className="bg-slate-50 rounded-xl p-3 border border-slate-100/80">
                            <p className="text-[10px] font-semibold text-slate-500 mb-2 uppercase tracking-wider">响应指标</p>
                            <div className="grid grid-cols-2 gap-2">
                              {Object.entries(bc.response).map(([k, v]) => (
                                <div key={k} className="flex items-center justify-between bg-white rounded-lg px-2.5 py-1.5 border border-slate-100/60">
                                  <span className="text-[10px] text-slate-500">{k}</span>
                                  <span className="text-[11px] font-mono font-semibold">{typeof v === 'number' ? (v * 100).toFixed(1) + '%' : v}</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-8 text-slate-400">
              <AlertTriangle size={28} className="text-brand-200 mb-2" />
              <p className="text-sm">点击「查看 Bad Case」获取得分最低的题目</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

/* ── 子组件：分数标签 ── */
function ScoreBadge({ label, value }: { label: string; value: number }) {
  const color = value >= 0.8 ? 'text-emerald-700 bg-emerald-50 border-emerald-200'
    : value >= 0.6 ? 'text-amber-700 bg-amber-50 border-amber-200'
    : value >= 0.4 ? 'text-orange-700 bg-orange-50 border-orange-200'
    : 'text-red-600 bg-red-50 border-red-200'
  return (
    <span className={`inline-flex items-center gap-1.5 text-[10px] font-semibold font-mono px-2 py-1 rounded-lg border ${color}`}>
      <span className="lowercase">{label}:</span>
      {(value * 100).toFixed(1)}%
    </span>
  )
}

/* ── 子组件：分数数值 ── */
function ScoreValue({ value }: { value: number }) {
  const color = value >= 0.8 ? 'text-emerald-600' : value >= 0.6 ? 'text-amber-600' : value >= 0.4 ? 'text-orange-600' : 'text-red-500'
  return (
    <span className={`text-sm font-bold font-mono ${color}`}>
      {(value * 100).toFixed(1)}%
    </span>
  )
}

/* ── 子组件：指标行（带进度条） ── */
function MetricRow({ label, value }: { label: string; value: number }) {
  const barGradient = value >= 0.8
    ? 'linear-gradient(90deg, #10b981, #059669)'
    : value >= 0.6
      ? 'linear-gradient(90deg, #f59e0b, #d97706)'
      : value >= 0.4
        ? 'linear-gradient(90deg, #f97316, #ea580c)'
        : 'linear-gradient(90deg, #ef4444, #dc2626)'
  const valColor = value >= 0.8 ? 'text-emerald-600'
    : value >= 0.6 ? 'text-amber-600'
    : value >= 0.4 ? 'text-orange-600'
    : 'text-red-500'

  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <span className="text-[11px] text-slate-600">{label}</span>
        <span className={`text-[11px] font-bold font-mono ${valColor}`}>{(value * 100).toFixed(1)}%</span>
      </div>
      <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
        <div className="h-full rounded-full transition-all duration-500" style={{ width: `${value * 100}%`, background: barGradient }} />
      </div>
    </div>
  )
}
