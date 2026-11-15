/**
 * TTS provider choices shared by SettingsPanel and OnboardingWizard.
 * Lives outside the component files so react-refresh stays happy.
 */

export type TtsProviderChoice =
  | 'melotts_onnx'
  | 'qwen3_tts'
  | 'openai'
  | 'huggingface_inference'
  | 'pyttsx3'

export const TTS_PROVIDER_OPTIONS: { value: TtsProviderChoice; label: string }[] = [
  { value: 'melotts_onnx', label: 'Kurisu voice (local, no API key)' },
  { value: 'openai', label: 'OpenAI' },
  { value: 'huggingface_inference', label: 'Hugging Face Space' },
  { value: 'pyttsx3', label: 'System voice (pyttsx3)' },
]

// Only offered when the backend is already running it (the GPU desktop
// build) — selecting it in a CPU build would just break the voice.
export const QWEN_TTS_OPTION: { value: TtsProviderChoice; label: string } = {
  value: 'qwen3_tts',
  label: 'Kurisu cloned voice (GPU)',
}

export const normalizeTtsProvider = (value: string): TtsProviderChoice => {
  // qwen3_tts is the GPU build's cloned voice. It MUST survive normalization:
  // mapping it to a fallback here meant any settings save silently downgraded
  // the packaged GPU voice to pyttsx3 (which isn't even bundled).
  if (
    value === 'melotts_onnx' ||
    value === 'qwen3_tts' ||
    value === 'openai' ||
    value === 'huggingface_inference'
  )
    return value
  return 'pyttsx3'
}
