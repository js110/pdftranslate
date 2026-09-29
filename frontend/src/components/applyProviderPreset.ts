import { getPreset } from '../providers'
import type { ProviderPresetKey } from '../providers'

type IdSetter = (value: string) => void

export function applyProviderPreset(
  presetKey: ProviderPresetKey,
  setId: IdSetter,
  setModel: IdSetter,
  setBaseUrl: IdSetter,
) {
  const preset = getPreset(presetKey)
  if (!preset || preset.key === 'custom') return
  setId(preset.id)
  setModel(preset.model)
  setBaseUrl(preset.baseUrl)
}
