import { useState, useEffect, useRef } from 'react'
import { BarChart3, Play, FileBarChart, Info, ChevronDown, ChevronUp, Search, Calculator, GitCompare, Eye, AlertTriangle, Activity, RotateCcw, FastForward, RefreshCw, ListChecks, Square, Trash2, Gauge, Layers, ArrowLeftRight, Sparkles, TrendingUp } from 'lucide-react'
import { getEvaluationReport, getObservability, getBadCases, createPersistentTask, startPersistentTask, resumePersistentTask, restartPersistentTask, retryFailedPersistentTask, forceStopPersistentTask, deletePersistentTask, listPersistentTasks, getPersistentProgress, getPersistentReport, type EvalComparison, type Observability, type BadCase, type PersistentTask, type PersistentProgress, type PersistentReport } from '../api'

const STRATEGY_LABELS: Record<string, string> = {
  vector: '纯向量检索',
  hybrid: '混合检索',
  reranked: '重排序检索',
  crag: 'CRAG纠错(已移除)',
  simple: '简单链路',
  agent: 'Agentic RAG',
}

const STRATEGY_COLORS: Record<string, string> = {
  vector: 'text-brand-600 bg-brand-50 border-brand-200',
  hybrid: 'text-accent-600 bg-accent-50 border-accent-200',
  reranked: 'text-cyan-600 bg-cyan-50 border-cyan-100',
  crag: 'text-amber-600 bg-amber-50 border-amber-200 opacity-50',
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
  const [evalSubset, setEvalSubset] = useState<string>('smoke')
  const [ragMode, setRagMode] = useState<string>('')
  const [questionType, setQuestionType] = useState<string>('')
  const [difficulty, setDifficulty] = useState<string>('')
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

  const handleRefreshAll = () => {
    listPersistentTasks().then(r => setPTasks(r.tasks)).catch(() => {})
    getEvaluationReport().then(setComparison).catch(() => {})
    getObservability().then(setObservability).catch(() => {})
    getBadCases().then(setBadCases).catch(() => {})
  }

  const metricLabel = (key: string) => {
    const labels: Record<string, string> = {
      faithfulness: '忠实度',
    }
    return labels[key] || key
  }

  // 核心指标：统一从 observability 提取（包含检索+生成全部5个指标）
  const coreMetrics = (() => {
    const m: { label: string; value: number; color: string }[] = []
    if (observability && !observability.message) {
      const rq = observability.retrieval_quality as Record<string, number>
      const gq = observability.generation_quality as Record<string, number>
      if (rq['precision@1'] != null) m.push({ label: '精确率 P@1', value: rq['precision@1'], color: 'text-cyan-600' })
      if (rq['recall@5'] != null) m.push({ label: '召回率 R@5', value: rq['recall@5'], color: 'text-amber-600' })
      if (gq['faithfulness'] != null) m.push({ label: '忠实度', value: gq['faithfulness'], color: 'text-emerald-600' })
      if (gq['hallucination_rate'] != null) m.push({ label: '幻觉率', value: gq['hallucination_rate'], color: 'text-red-500' })
      if (gq['completeness'] != null) m.push({ label: '完整性', value: gq['completeness'], color: 'text-violet-600' })
    }
    return m
  })()

  return (
    <div className="max-w-5xl mx-auto p-5 lg:p-6 space-y-5 overflow-y-auto h-full pb-8">
      {/* ── 页面标题 ── */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-brand-500 to-accent-500 flex items-center justify-center shadow-sm shadow-brand-200/40">
            <BarChart3 size={20} className="text-white" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-gradient-brand">评估系统</h2>
            <p className="text-xs text-slate-400 mt-0.5">任务管理 · 核心指标 · 深度分析</p>
          </div>
        </div>
        <button
          onClick={handleRefreshAll}
          className="inline-flex items-center gap-1.5 rounded-lg btn-brand px-3.5 py-2 text-xs transition-all duration-200"
        >
          <RefreshCw size={14} />
          刷新全部
        </button>
      </div>

      {/* ══════════════════════════════════════════════════════
          第一层：评估任务管理
      ══════════════════════════════════════════════════════ */}
      <div className="bg-white rounded-xl border border-slate-200/80 shadow-card overflow-hidden">
        <div className="px-5 py-4 border-b border-brand-100/50 flex items-center justify-between bg-brand-50/30">
          <h3 className="font-semibold text-slate-700 flex items-center gap-2 text-sm">
            <ListChecks size={16} className="text-accent-500" />
            评估任务
          </h3>
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
            <option value="simple">简单链路</option>
            <option value="crag" disabled>CRAG纠错(已移除)</option>
            <option value="agent">Agentic RAG</option>
          </select>
          <select
            value={evalSubset}
            onChange={(e) => setEvalSubset(e.target.value)}
            className="rounded-lg border border-slate-200 bg-white px-2.5 py-1.5 text-xs text-slate-700 focus:outline-none focus:ring-2 focus:ring-brand-400/30 focus:border-brand-400"
          >
            <option value="smoke">Smoke(5题/1.5min)</option>
            <option value="full">Full(20题/5-8min)</option>
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
          <select
            value={difficulty}
            onChange={(e) => setDifficulty(e.target.value)}
            className="rounded-lg border border-slate-200 bg-white px-2.5 py-1.5 text-xs text-slate-700 focus:outline-none focus:ring-2 focus:ring-brand-400/30 focus:border-brand-400"
          >
            <option value="">全部难度</option>
            <option value="easy">Easy</option>
            <option value="medium">Medium</option>
            <option value="hard">Hard</option>
          </select>
          <button
            onClick={async () => {
              try {
                setPRunning(true)
                const { task_id } = await createPersistentTask(
                  ragMode || undefined, undefined, questionType || undefined, undefined,
                  evalSubset || undefined, difficulty || undefined,
                )
                await startPersistentTask(task_id)
                startPPolling(task_id)
                listPersistentTasks().then(r => setPTasks(r.tasks)).catch(() => {})
              } catch (err) { setPRunning(false); alert(`创建任务失败: ${(err as Error).message}`) }
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
                const total = task.total_questions ?? 0
                const pct = typeof total === 'number' && total > 0 ? Math.round(done / total * 100) : 0
                return (
                  <div key={task.task_id} className="border border-slate-100/80 rounded-xl p-4 bg-white hover:border-brand-200 transition-colors shadow-sm">
                    {/* 任务标题行：状态+策略+操作按钮 */}
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
                      {/* 统一操作入口：高频按钮集中在此 */}
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
                              } catch (err) { setPRunning(false); alert(`重新评估失败: ${(err as Error).message}`) }
                            }}
                            className="inline-flex items-center gap-1 rounded-lg bg-amber-500 px-2.5 py-1.5 text-[11px] font-medium text-white hover:bg-amber-600 transition-all duration-200 shadow-sm"
                          >
                            <RotateCcw size={11} /> 重评
                          </button>
                        )}
                        {task.status === 'completed' && (
                          <>
                            <button
                              onClick={() => getPersistentReport(task.task_id).then(setPReport).catch(() => {})}
                              className="inline-flex items-center gap-1 rounded-lg bg-green-500 px-2.5 py-1.5 text-[11px] font-medium text-white hover:bg-green-600 transition-all duration-200 shadow-sm"
                            >
                              <FileBarChart size={11} /> 报告
                            </button>
                            <button
                              onClick={() => { handleCompare(); getObservability().then(setObservability).catch(() => {}); getBadCases().then(setBadCases).catch(() => {}) }}
                              className="inline-flex items-center gap-1 rounded-lg bg-brand-500 px-2.5 py-1.5 text-[11px] font-medium text-white hover:bg-brand-600 transition-all duration-200 shadow-sm"
                            >
                              <Eye size={11} /> 分析
                            </button>
                          </>
                        )}
                        {task.status !== 'running' && task.status !== 'scoring' && fail > 0 && (
                          <button
                            onClick={async () => {
                              try {
                                setPRunning(true)
                                await retryFailedPersistentTask(task.task_id)
                                startPPolling(task.task_id)
                              } catch (err) { setPRunning(false); alert(`重试失败: ${(err as Error).message}`) }
                            }}
                            className="inline-flex items-center gap-1 rounded-lg bg-red-500 px-2.5 py-1.5 text-[11px] font-medium text-white hover:bg-red-600 transition-all duration-200 shadow-sm"
                          >
                            <RefreshCw size={11} /> 重试({fail})
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
      </div>

      {/* ══════════════════════════════════════════════════════
          第二层：核心评估指标卡片（5个核心指标 + 概要统计）
      ══════════════════════════════════════════════════════ */}
      {(coreMetrics.length > 0 || (pReport && pReport.summary)) && (
        <div className="bg-white rounded-xl border border-slate-200/80 shadow-card overflow-hidden">
          <div className="px-5 py-4 border-b border-emerald-100/50 flex items-center justify-between bg-emerald-50/30">
            <div className="flex items-center gap-2">
              <TrendingUp size={16} className="text-emerald-500" />
              <h3 className="font-semibold text-slate-700 text-sm">核心评估指标</h3>
            </div>
            {coreMetrics.length === 0 && (
              <button
                onClick={() => getObservability().then(setObservability).catch(() => {})}
                className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-500 px-3 py-1.5 text-xs font-medium text-white hover:bg-emerald-600 transition-all duration-200"
              >
                <Eye size={13} />
                加载指标
              </button>
            )}
          </div>

          <div className="px-5 py-5">
            {/* 概要统计 */}
            {pReport && pReport.summary && (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
                {[
                  { label: '总题数', value: pReport.summary.total, color: 'text-slate-700' },
                  { label: '成功', value: pReport.summary.success, color: 'text-emerald-600' },
                  { label: '失败', value: pReport.summary.failed, color: 'text-red-500' },
                  { label: '成功率', value: `${pReport.summary.success_rate}%`, color: 'text-brand-600' },
                ].map((item) => (
                  <div key={item.label} className="text-center bg-slate-50/80 rounded-lg p-2.5 border border-slate-100">
                    <p className={`text-lg font-bold ${item.color}`}>{item.value}</p>
                    <p className="text-[10px] text-gray-500">{item.label}</p>
                  </div>
                ))}
              </div>
            )}

            {/* 核心指标卡片（去重：faithfulness/P@5等只在这里出现一次） */}
            {coreMetrics.length > 0 && (
              <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                {coreMetrics.map((m) => (
                  <div key={m.label} className="text-center bg-gradient-to-b from-white to-slate-50/50 rounded-xl p-3 border border-slate-100 shadow-sm">
                    <p className="text-[10px] text-slate-500 mb-1.5">{m.label}</p>
                    <p className={`text-xl font-bold font-mono ${m.color}`}>{(m.value * 100).toFixed(1)}%</p>
                    <div className="mt-2 h-1.5 bg-slate-100 rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all duration-500"
                        style={{
                          width: `${m.value * 100}%`,
                          background: m.value >= 0.8
                            ? 'linear-gradient(90deg, #10b981, #059669)'
                            : m.value >= 0.6
                              ? 'linear-gradient(90deg, #f59e0b, #d97706)'
                              : m.value >= 0.4
                                ? 'linear-gradient(90deg, #f97316, #ea580c)'
                                : 'linear-gradient(90deg, #ef4444, #dc2626)'
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* 失败条目 */}
            {pReport?.failed_items && pReport.failed_items.length > 0 && (
              <div className="mt-4">
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
        </div>
      )}

      {/* ══════════════════════════════════════════════════════
          第三层：深度分析（业务指标 + 策略对比 + Bad Case）
      ══════════════════════════════════════════════════════ */}

      {/* ── 业务指标面板 ── */}
      <div className="bg-white rounded-xl border border-slate-200/80 shadow-card overflow-hidden">
        <div className="px-5 py-4 border-b border-cyan-100/50 flex items-center justify-between bg-cyan-50/30">
          <h3 className="font-semibold text-slate-700 flex items-center gap-2 text-sm">
            <Activity size={16} className="text-cyan-500" />
            业务指标
          </h3>
          <button
            onClick={() => getObservability().then(setObservability).catch(() => {})}
            className="inline-flex items-center gap-1.5 rounded-lg bg-cyan-500 px-3.5 py-2 text-xs font-medium text-white hover:bg-cyan-600 transition-all duration-200 shadow-sm"
          >
            <Activity size={14} />
            加载数据
          </button>
        </div>

        <div className="px-5 pb-5 pt-4">
          {observability && !observability.message ? (
            <div className="grid grid-cols-2 gap-4">
              <div className="rounded-xl border border-cyan-100/80 bg-gradient-to-b from-cyan-50/60 to-white p-4">
                <MetricRow label="问题解决率" value={observability.business.resolution_rate ?? 0} />
              </div>
              <div className="rounded-xl border border-cyan-100/80 bg-gradient-to-b from-cyan-50/60 to-white p-4">
                <MetricRow label="首次回答可用率" value={observability.business.first_answer_usability ?? 0} />
              </div>
            </div>
          ) : !observability ? (
            <div className="flex flex-col items-center justify-center py-8 text-slate-400">
              <Activity size={28} className="text-cyan-200 mb-2" />
              <p className="text-sm">点击「加载数据」获取业务指标</p>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-8 text-slate-400">
              <p className="text-sm">{observability.message}</p>
            </div>
          )}
        </div>
      </div>

      {/* ── 策略对比报告 ── */}
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
              <p className="font-semibold mb-0.5">Golden Set 双轨评估</p>
              <p><b>Smoke</b>(5题/1.5min)：快速验证核心检索链路与安全护栏，适合日常迭代。</p>
              <p><b>Full</b>(20题/5-8min)：完整评估，含计算推理/冷门法规/超纲拒答等长尾场景，适合系统验收。</p>
              <p className="mt-0.5">也可在后端运行 <code className="bg-brand-100/80 px-1.5 py-0.5 rounded text-[10px] font-mono">python run_eval.py --subset smoke</code> 或 <code className="bg-brand-100/80 px-1.5 py-0.5 rounded text-[10px] font-mono">python run_eval.py --subset full</code>。</p>
            </div>
          </div>
        </div>

        <div className="px-5 pb-5">
          {comparison && comparison.results && comparison.metrics && Object.keys(comparison.results).length > 0 ? (
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
