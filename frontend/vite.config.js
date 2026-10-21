import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { createReadStream, existsSync, statSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { join, normalize, resolve, sep } from 'node:path'

const here = resolve(fileURLToPath(new URL('.', import.meta.url)))
const repoRoot = resolve(here, '..')
const experimentAssets = resolve(repoRoot, 'experiments', 'browser-feasibility', 'browser-yolo-test', 'public')
const localModelPrefix = '/browser-models/scene-graph-v1/'

// Development/preview-only bridge to the immutable E1 artifacts. It avoids a
// second 231 MB model copy under frontend/public and is not included in the JS
// bundle. Production hosts the same versioned tree at VITE_BROWSER_MODEL_BASE_URL.
function localBrowserModelAssets() {
  function serve(req, res, next) {
    if (!req.url?.startsWith(localModelPrefix)) return next()
    const relativePath = decodeURIComponent(req.url.slice(localModelPrefix.length).split('?')[0])
    const filePath = normalize(join(experimentAssets, relativePath))
    if (!filePath.startsWith(`${experimentAssets}${sep}`) || !existsSync(filePath) || !statSync(filePath).isFile()) {
      res.statusCode = 404
      res.end('Browser model asset not found')
      return
    }
    res.setHeader('Cache-Control', 'no-cache')
    if (filePath.endsWith('.onnx')) res.setHeader('Content-Type', 'application/octet-stream')
    else if (filePath.endsWith('.wasm')) res.setHeader('Content-Type', 'application/wasm')
    else if (filePath.endsWith('.mjs')) res.setHeader('Content-Type', 'text/javascript')
    else if (filePath.endsWith('.json')) res.setHeader('Content-Type', 'application/json')
    createReadStream(filePath).pipe(res)
  }
  return {
    name: 'local-browser-model-assets',
    configureServer(server) { server.middlewares.use(serve) },
    configurePreviewServer(server) { server.middlewares.use(serve) },
  }
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), localBrowserModelAssets()],
  resolve: {
    // Select ONNX Runtime's external-WASM entry point. Runtime files are served
    // from the versioned browser model asset prefix, not copied into dist/.
    conditions: ['onnxruntime-web-use-extern-wasm'],
  },
  server: {
    host: '127.0.0.1',
    port: 5173,
  },
})
