export type ProviderPresetKey = 'deepseek' | 'tencent' | 'aliyun' | 'xiaomi' | 'custom'

export type ProviderPreset = {
  key: ProviderPresetKey
  label: string
  id: string
  model: string
  baseUrl: string
}

export const PROVIDER_PRESETS: ProviderPreset[] = [
  {
    key: 'deepseek',
    label: 'DeepSeek',
    id: 'deepseek-main',
    model: 'deepseek-chat',
    baseUrl: 'https://api.deepseek.com/v1',
  },
  {
    key: 'tencent',
    label: '腾讯混元',
    id: 'hunyuan-main',
    model: 'hunyuan-turbos-latest',
    baseUrl: 'https://api.hunyuan.cloud.tencent.com/v1',
  },
  {
    key: 'aliyun',
    label: '阿里通义千问',
    id: 'qwen-main',
    model: 'qwen-plus',
    baseUrl: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
  },
  {
    key: 'xiaomi',
    label: '小米 MiMo',
    id: 'mimo-main',
    model: 'mimo-v2.5-pro',
    baseUrl: 'https://token-plan-sgp.xiaomimimo.com/v1',
  },
  {
    key: 'custom',
    label: '自定义（OpenAI 兼容）',
    id: 'custom-main',
    model: '',
    baseUrl: '',
  },
]

export function inferPresetKey(baseUrl: string, model: string): ProviderPresetKey {
  const lowerUrl = baseUrl.toLowerCase()
  const lowerModel = model.toLowerCase()
  if (lowerUrl.includes('api.deepseek.com') || lowerModel.includes('deepseek')) return 'deepseek'
  if (lowerUrl.includes('hunyuan.cloud.tencent.com') || lowerModel.includes('hunyuan')) return 'tencent'
  if (lowerUrl.includes('dashscope.aliyuncs.com') || lowerModel.includes('qwen')) return 'aliyun'
  if (lowerUrl.includes('xiaomimimo.com') || lowerModel.includes('mimo')) return 'xiaomi'
  return 'custom'
}

export function getPreset(presetKey: ProviderPresetKey): ProviderPreset | undefined {
  return PROVIDER_PRESETS.find((item) => item.key === presetKey)
}

/** MiMo is a reasoning model: give its long chain a generous timeout. */
export function inferTimeout(presetKey: ProviderPresetKey): number {
  return presetKey === 'xiaomi' ? 120 : 60
}
