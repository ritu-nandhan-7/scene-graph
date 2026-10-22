import { createReadStream, existsSync, statSync } from 'node:fs'
import { createServer } from 'node:http'
import { dirname, join, normalize, resolve, sep } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const root = resolve(here, 'browser-yolo-test', 'public')
const prefix = '/browser-models/scene-graph-v1/'
const port = Number(process.env.F3A_ASSET_PORT || 4184)
const mime = { '.onnx': 'application/octet-stream', '.json': 'application/json', '.mjs': 'text/javascript', '.wasm': 'application/wasm' }

createServer((request, response) => {
  if (!request.url?.startsWith(prefix)) { response.writeHead(404).end('Not found'); return }
  const relative = decodeURIComponent(request.url.slice(prefix.length).split('?')[0])
  const file = normalize(join(root, relative))
  if (!file.startsWith(`${root}${sep}`) || !existsSync(file) || !statSync(file).isFile()) { response.writeHead(404, { 'Access-Control-Allow-Origin': '*' }).end('Asset not found'); return }
  const extension = file.slice(file.lastIndexOf('.'))
  response.writeHead(200, {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Cache-Control': 'public, max-age=31536000, immutable',
    'Content-Type': mime[extension] || 'application/octet-stream',
  })
  if (request.method === 'HEAD') { response.end(); return }
  createReadStream(file).pipe(response)
}).listen(port, '127.0.0.1', () => console.log(`F3A asset host: http://127.0.0.1:${port}${prefix}`))