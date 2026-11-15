!macro NSIS_HOOK_PREINSTALL
  !insertmacro CheckIfAppIsRunning "amadeus-desktop.exe" "Amadeus Kurisu"
  !insertmacro CheckIfAppIsRunning "amadeus-backend.exe" "${PRODUCTNAME} backend"
!macroend
