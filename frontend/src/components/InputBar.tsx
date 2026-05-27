import { useState, useRef, type FormEvent, type KeyboardEvent } from 'react'
import { Send, Sparkles } from 'lucide-react'

interface InputBarProps {
  onSend: (message: string) => void
  disabled?: boolean
}

export default function InputBar({ onSend, disabled }: InputBarProps) {
  const [input, setInput] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault()
    const trimmed = input.trim()
    if (!trimmed || disabled) return
    onSend(trimmed)
    setInput('')
    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  const adjustHeight = () => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 120) + 'px'
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="flex items-end gap-2.5 px-4 lg:px-6 py-3.5 border-t border-brand-100/40 bg-white/80 backdrop-blur-md"
    >
      <div className="flex-1 relative">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => {
            setInput(e.target.value)
            adjustHeight()
          }}
          onKeyDown={handleKeyDown}
          placeholder="输入劳动法相关问题，如“加班费怎么算”或“月薪8000加班10小时加班费多少”..."
          disabled={disabled}
          rows={1}
          className="w-full resize-none rounded-xl border border-slate-300/80 bg-slate-50/50 px-4 py-2.5 pr-10 text-sm focus:outline-none focus:ring-2 focus:ring-brand-400/30 focus:border-brand-400 focus:bg-white disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 placeholder:text-slate-400"
          style={{ maxHeight: '120px' }}
        />
        {!input.trim() && !disabled && (
          <div className="absolute right-3 bottom-2.5 pointer-events-none">
            <Sparkles size={16} className="text-brand-300" />
          </div>
        )}
      </div>
      <button
        type="submit"
        disabled={disabled || !input.trim()}
        className="flex-shrink-0 h-10 w-10 rounded-xl bg-gradient-to-br from-brand-500 to-accent-500 text-white flex items-center justify-center disabled:opacity-40 disabled:cursor-not-allowed transition-all duration-200 shadow-sm shadow-brand-200/50 hover:shadow-md hover:shadow-brand-300/50 active:scale-93"
      >
        <Send size={17} strokeWidth={1.8} />
      </button>
    </form>
  )
}
