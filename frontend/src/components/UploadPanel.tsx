import { useState } from 'react'
import type { FormEvent } from 'react'

type Props = {
  file: File | null
  setFile: (file: File | null) => void
  loading: boolean
  onUpload: (event: FormEvent) => void
}

export function UploadPanel({ file, setFile, loading, onUpload }: Props) {
  const [dragging, setDragging] = useState(false)

  return (
    <section className="upload-stage">
      <div className="upload-panel panel">
        <h2>{'\u4e0a\u4f20\u79d1\u7814 PDF'}</h2>
        <p className="upload-desc">{'\u5355\u7bc7\u8bba\u6587\u5728\u7ebf\u53cc\u680f\u7ffb\u8bd1\uff0c\u652f\u6301\u524d 3 \u9875\u4f18\u5148\u53ef\u8bfb'}</p>
        <form onSubmit={onUpload} className="upload-form">
          <label
            className={`file-picker${dragging ? ' dragging' : ''}`}
            onDragOver={(e) => {
              e.preventDefault()
              setDragging(true)
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault()
              setDragging(false)
              const dropped = e.dataTransfer.files[0]
              if (dropped && dropped.type === 'application/pdf') {
                setFile(dropped)
              }
            }}
          >
            <input
              type="file"
              accept="application/pdf"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
            <span>{file ? file.name : '\u70b9\u51fb\u9009\u62e9 PDF \u6587\u4ef6'}</span>
          </label>
          <button type="submit" disabled={loading || !file}>
            {loading ? '\u4e0a\u4f20\u4e2d...' : '\u4e0a\u4f20 PDF \u5e76\u521b\u5efa\u4f1a\u8bdd'}
          </button>
        </form>
        <p className="upload-hint">{'\u5efa\u8bae 30MB \u4ee5\u5185\uff0c\u4f1a\u8bdd\u5173\u95ed\u540e\u81ea\u52a8\u6e05\u7406'}</p>
      </div>
    </section>
  )
}
