import { enablePetDefaultAfterOnboarding } from './petDefault'

const ONBOARDING_DISMISSED_KEY = 'kurisu_onboarding_dismissed'

export const markOnboardingDismissed = () => {
  if (typeof window === 'undefined') return
  window.localStorage.setItem(ONBOARDING_DISMISSED_KEY, 'true')
  // Desktop installs live as the pet from the next launch on (the chat
  // window stays one tray click away); no-op in browsers.
  enablePetDefaultAfterOnboarding()
}

export const isOnboardingDismissed = () => {
  if (typeof window === 'undefined') return true
  return window.localStorage.getItem(ONBOARDING_DISMISSED_KEY) === 'true'
}
