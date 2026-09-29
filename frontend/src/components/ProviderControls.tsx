import { applyProviderPreset } from './applyProviderPreset'
import { PROVIDER_PRESETS } from '../providers'
import type { ProviderPresetKey } from '../providers'

type Props = {
  primaryPreset: ProviderPresetKey
  setPrimaryPreset: (value: ProviderPresetKey) => void
  primaryId: string
  setPrimaryId: (value: string) => void
  primaryModel: string
  setPrimaryModel: (value: string) => void
  primaryBaseUrl: string
  setPrimaryBaseUrl: (value: string) => void
  primaryKey: string
  setPrimaryKey: (value: string) => void
}

export function ProviderControls(props: Props) {
  const {
    primaryPreset,
    setPrimaryPreset,
    primaryId,
    setPrimaryId,
    primaryModel,
    setPrimaryModel,
    primaryBaseUrl,
    setPrimaryBaseUrl,
    primaryKey,
    setPrimaryKey,
  } = props

  return (
    <div className="controls-row">
      <label>
        {'主模型提供商'}
        <select
          value={primaryPreset}
          onChange={(e) => {
            const presetKey = e.target.value as ProviderPresetKey
            setPrimaryPreset(presetKey)
            applyProviderPreset(presetKey, setPrimaryId, setPrimaryModel, setPrimaryBaseUrl)
          }}
        >
          {PROVIDER_PRESETS.map((preset) => (
            <option key={`primary-${preset.key}`} value={preset.key}>
              {preset.label}
            </option>
          ))}
        </select>
      </label>
      <label>
        {'\u4e3b\u6a21\u578b ID'}
        <input value={primaryId} onChange={(e) => setPrimaryId(e.target.value)} />
      </label>
      <label>
        {'\u4e3b\u6a21\u578b\u540d\u79f0'}
        <input value={primaryModel} onChange={(e) => setPrimaryModel(e.target.value)} />
      </label>
      <label>
        {'\u4e3b\u6a21\u578b Base URL'}
        <input value={primaryBaseUrl} onChange={(e) => setPrimaryBaseUrl(e.target.value)} />
      </label>
      <label>
        {'\u4e3b\u6a21\u578b API Key'}
        <input type="password" value={primaryKey} onChange={(e) => setPrimaryKey(e.target.value)} />
      </label>
    </div>
  )
}
