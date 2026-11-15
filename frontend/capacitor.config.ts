import type { CapacitorConfig } from '@capacitor/cli'

const config: CapacitorConfig = {
  appId: 'com.amadeus.kurisu',
  appName: 'Amadeus',
  webDir: 'dist',
  bundledWebRuntime: false,
  server: {
    // The app shell is served from the secure capacitor origin
    // (https://localhost); the self-hosted backend is usually plain HTTP on
    // the LAN, which would otherwise be blocked as mixed content.
    androidScheme: 'https',
    cleartext: true,
  },
  android: {
    allowMixedContent: true,
  },
}

export default config
