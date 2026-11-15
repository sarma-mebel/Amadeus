import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { createReadStream, existsSync, statSync } from 'node:fs'
import { extname, join, normalize, resolve, sep } from 'node:path'
import { fileURLToPath } from 'node:url'
import type { Plugin, ViteDevServer } from 'vite'

const frontendDir = fileURLToPath(new URL('.', import.meta.url))
const repoStaticDir = resolve(frontendDir, '..', 'static')

const mimeTypes: Record<string, string> = {
  '.glb': 'model/gltf-binary',
  '.json': 'application/json',
  '.png': 'image/png',
  '.vrm': 'model/gltf-binary',
  '.vrma': 'model/gltf-binary',
  '.webp': 'image/webp',
}

const localStaticAssets = (): Plugin => ({
  name: 'local-static-assets',
  configureServer(server: ViteDevServer) {
    server.middlewares.use('/static', (request, response, next) => {
      const rawUrl = request.url?.split('?')[0] ?? '/'
      const relativePath = decodeURIComponent(rawUrl).replace(/^\/+/, '')
      const filePath = normalize(join(repoStaticDir, relativePath))
      if (!filePath.startsWith(repoStaticDir + sep) && filePath !== repoStaticDir) {
        response.statusCode = 403
        response.end('Forbidden')
        return
      }
      if (!existsSync(filePath) || !statSync(filePath).isFile()) {
        next()
        return
      }
      const fileStat = statSync(filePath)
      response.setHeader('Content-Type', mimeTypes[extname(filePath).toLowerCase()] ?? 'application/octet-stream')
      response.setHeader('Content-Length', fileStat.size)
      if (request.method === 'HEAD') {
        response.end()
        return
      }
      createReadStream(filePath)
        .on('error', () => {
          if (!response.headersSent) {
            response.statusCode = 500
          }
          response.end()
        })
        .pipe(response)
    })
  },
})

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), localStaticAssets()],
  server: {
    watch: {
      // src-tauri holds the bundled PyInstaller backend (thousands of torch
      // files, plus amadeus-backend.exe which is locked while running) —
      // watching it crashes the dev server with EBUSY on Windows.
      ignored: ['**/src-tauri/**'],
    },
    proxy: {
      // 127.0.0.1, not localhost: on Windows, Node resolves localhost to
      // ::1 first while the backend binds IPv4 only -> ECONNREFUSED.
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      // The connection screen probes backend readiness via /healthz.
      '/healthz': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          // three.js + @pixiv/three-vrm are the bulk of the bundle and are
          // only needed once the lazy-loaded VRMViewer chunk mounts; keeping
          // them in their own chunk means the chat UI's own code doesn't
          // wait on the 3D stack to finish downloading.
          'vendor-three': ['three', '@pixiv/three-vrm', '@pixiv/three-vrm-animation'],
        },
      },
    },
  },
})
