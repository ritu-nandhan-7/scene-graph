import { chromium } from 'playwright'
import { existsSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const repo = join(here, '..')
const appUrl = process.env.F3A_APP_URL || 'http://127.0.0.1:4173/'
const assetOrigin = process.env.F3A_ASSET_ORIGIN || 'http://127.0.0.1:4184'
const fixtures = [
  ['few-100.jpg', 2, 1], ['fixture-1.jpg', 10, 15], ['moderate-107899.jpg', 14, 23], ['many-107900.jpg', 3, 3], ['overlap-107901.jpg', 7, 12],
]
const browser = await chromium.launch({ channel: 'chrome' }).catch(() => chromium.launch())
const context = await browser.newContext({ viewport: { width: 1600, height: 1000 } })
const page = await context.newPage()
const errors = [], requests = [], timings = [], frameGaps = []
page.on('pageerror', (error) => errors.push(String(error)))
page.on('request', (request) => { if (/browser-models|\/api\/analyze|:8000\/analyze/.test(request.url())) requests.push(request.url()) })
await page.addInitScript(() => {
  window.__f3aTimings = []
  window.addEventListener('scene-graph-inference-timing', (event) => window.__f3aTimings.push(event.detail))
  let last = performance.now()
  const tick = (now) => { window.__f3aFrameGaps = [...(window.__f3aFrameGaps || []), now - last]; last = now; requestAnimationFrame(tick) }
  requestAnimationFrame(tick)
})
await page.goto(appUrl, { waitUntil: 'networkidle' })
const results = []
for (const [fixture, expectedObjects, expectedRelationships] of fixtures) {
  const file = join(repo, 'experiments', 'browser-feasibility', 'browser-yolo-test', 'public', 'e1', 'images', fixture)
  if (!existsSync(file)) throw new Error(`Missing fixture ${file}`)
  await page.setInputFiles('input[type=file]', file)
  await page.locator('button:has-text("Run analysis")').click()
  const started = Date.now()
  await page.waitForFunction(() => document.querySelectorAll('.bbox').length > 0 || document.body.textContent.includes('No objects detected'), null, { timeout: 180000 })
  await page.waitForFunction(() => !document.querySelector('.loading-card'), null, { timeout: 180000 })
  const values = await page.evaluate(() => ({ objects: document.querySelectorAll('.bbox').length, nodes: document.querySelectorAll('.react-flow__node').length, relationships: document.querySelectorAll('.edge-label').length, timings: window.__f3aTimings.splice(0), gaps: window.__f3aFrameGaps.splice(0) }))
  timings.push(...values.timings); frameGaps.push(...values.gaps)
  results.push({ fixture, expectedObjects, expectedRelationships, ...values, elapsedMs: Date.now() - started, passed: values.objects === expectedObjects && values.relationships === expectedRelationships && values.nodes === expectedObjects })
}
const headers = await Promise.all([...new Set(requests.filter((url) => url.startsWith(assetOrigin)))].map(async (url) => { const response = await fetch(url, { method: 'HEAD' }); return { url, status: response.status, cors: response.headers.get('access-control-allow-origin'), mime: response.headers.get('content-type'), cache: response.headers.get('cache-control') } }))
const report = { appUrl, assetOrigin, results, errors, requests, headers, maxAnimationFrameGapMs: Math.max(...frameGaps), timings, fastApiRequests: requests.filter((url) => /\/api\/analyze|:8000\/analyze/.test(url)) }
console.log(JSON.stringify(report, null, 2))
await browser.close()
if (errors.length || report.fastApiRequests.length || results.some((result) => !result.passed)) process.exitCode = 1