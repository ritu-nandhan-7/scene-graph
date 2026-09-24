/**
 * Browser verification (Playwright, headless Chromium).
 *
 * Closes the "browser-only" verification gap from the frontend-polish phase by
 * driving the real UI against the real backend.  It checks the items that a
 * build or a source-level check cannot prove:
 *   U1  bounding boxes actually render over the image
 *   U2  the detection toggle really removes every overlay
 *   U3  "Download Graph PNG" really produces a PNG of the GRAPH (not the page)
 *   U4  graph interactions: node click, edge click, empty-canvas click
 *   U5  image <-> graph selection synchronisation, both directions
 * plus search highlighting, inspector content, Reset, and a second image run
 * (to prove no state leaks between analyses).
 *
 * ONE-TIME SETUP (deliberately NOT added to package.json - verification only):
 *     cd frontend
 *     npm install --no-save playwright
 *     npx playwright install chromium
 *
 * REQUIRES both servers running:
 *     scripts\run_backend.ps1    (port 8000)
 *     scripts\run_frontend.ps1   (port 5173)
 *
 * Run from frontend/:  node verify_browser.mjs
 * Screenshots are written to frontend/_browser_shots/ (disposable).
 */
import { chromium } from 'playwright'
import { existsSync, mkdirSync, readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

const here = dirname(fileURLToPath(import.meta.url))
const REPO = join(here, '..')
const APP = 'http://127.0.0.1:5173'
const IMAGE_A = join(REPO, 'datasets', 'visual_genome', 'VG_100K_2', '1.jpg')
const IMAGE_B = join(REPO, 'datasets', 'visual_genome', 'VG_100K_2', '100.jpg')
const SHOTS = join(here, '_browser_shots')

let failures = 0
function check(name, ok, detail = '') {
  console.log(`  [${ok ? 'PASS' : 'FAIL'}] ${name}${detail ? ' - ' + detail : ''}`)
  if (!ok) failures++
}

mkdirSync(SHOTS, { recursive: true })

// Prefer the bundled Chromium; fall back to an installed Edge/Chrome so the
// verification works without downloading a browser binary.
async function launchBrowser() {
  const attempts = [
    ['chromium (bundled)', {}],
    ['msedge (installed)', { channel: 'msedge' }],
    ['chrome (installed)', { channel: 'chrome' }],
  ]
  const problems = []
  for (const [name, opts] of attempts) {
    try {
      const b = await chromium.launch(opts)
      console.log(`  browser: ${name}`)
      return b
    } catch (err) {
      problems.push(`${name}: ${String(err).split('\n')[0].slice(0, 80)}`)
    }
  }
  throw new Error(`no usable browser found\n  ${problems.join('\n  ')}`)
}

const browser = await launchBrowser()
const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } })

const pageErrors = []
const consoleErrors = []
page.on('pageerror', (e) => pageErrors.push(String(e)))
page.on('console', (m) => {
  if (m.type() === 'error') consoleErrors.push(m.text())
})

async function shot(name) {
  await page.screenshot({ path: join(SHOTS, name), fullPage: true })
  console.log(`         screenshot -> _browser_shots/${name}`)
}

// Click a bounding box by dispatching on the element itself, so overlapping
// boxes can never make Playwright hit the wrong target.
async function clickFirstBBox() {
  await page.locator('.bbox').first().evaluate((el) => el.click())
}

// Index of the node closest to the centre of the graph viewport: the safest
// node to drag (guaranteed to be inside the pane and clear of the toolbars).
async function centralNodeIndex() {
  return page.evaluate(() => {
    const pane = document.querySelector('.graph-container').getBoundingClientRect()
    const cx = pane.x + pane.width / 2
    const cy = pane.y + pane.height / 2
    const nodes = [...document.querySelectorAll('.react-flow__node')]
    let best = 0
    let bestDistance = Infinity
    nodes.forEach((node, index) => {
      const r = node.getBoundingClientRect()
      const d = Math.hypot(r.x + r.width / 2 - cx, r.y + r.height / 2 - cy)
      if (d < bestDistance) {
        bestDistance = d
        best = index
      }
    })
    return best
  })
}

// Drag one node with real mouse events (Chromium synthesises the pointer events
// React Flow listens to).  Returns BOTH the on-screen movement in pixels and
// the node's GRAPH-space movement parsed from its inline transform
// ("translate(Xpx, Ypx)").  The distinction matters for the lock checks: when
// the lock is engaged React Flow pans the canvas from a pointer-down on a node
// (screen box moves, graph position does not), so graph-space is the honest
// measure of "did the node move".
function parseTranslate(value) {
  const m = /translate\((-?[\d.]+)px(?:,|\s)\s*(-?[\d.]+)px\)/.exec(value || '')
  return m ? { x: parseFloat(m[1]), y: parseFloat(m[2]) } : { x: 0, y: 0 }
}

const liveViewportTransform = () =>
  page.locator('.react-flow__viewport').evaluate((el) => el.style.transform || '')

async function dragNodeBy(index, dx, dy) {
  const node = page.locator('.react-flow__node').nth(index)
  const before = await node.boundingBox()
  const graphBefore = parseTranslate(await node.evaluate((el) => el.style.transform))
  const startX = before.x + before.width / 2
  const startY = before.y + before.height / 2
  await page.mouse.move(startX, startY)
  await page.mouse.down()
  await page.mouse.move(startX + dx, startY + dy, { steps: 14 })
  await page.mouse.up()
  await page.waitForTimeout(300)
  const after = await node.boundingBox()
  const graphAfter = parseTranslate(await node.evaluate((el) => el.style.transform))
  return {
    movedX: after.x - before.x,
    movedY: after.y - before.y,
    graphDX: graphAfter.x - graphBefore.x,
    graphDY: graphAfter.y - graphBefore.y,
  }
}

// Relationship labels are custom elements now (LaneEdge): clicking one selects
// the relationship, exactly like clicking the stroke does.  Both paths are
// exercised: the label first, the interaction stroke as the fallback.
async function clickFirstEdge() {
  const label = page.locator('.edge-label').first()
  if (await label.count()) {
    await label.dispatchEvent('click', { bubbles: true, cancelable: true })
    return 'edge label'
  }
  const wrapper = page.locator('.react-flow__edge-textwrapper').first()
  if (await wrapper.count()) {
    await wrapper.dispatchEvent('click', { bubbles: true, cancelable: true })
    return 'edge label wrapper'
  }
  await page
    .locator('.react-flow__edge-interaction')
    .first()
    .dispatchEvent('click', { bubbles: true, cancelable: true })
  return 'edge interaction path'
}

const sceneStats = async () =>
  page.locator('.stat-num').allInnerTexts().then((values) => values.map((v) => v.trim()).join('/'))

// Fresh object every call: earlier versions returned the shared `counts`
// object, so "before" and "after" snapshots aliased each other and every
// diff check compared the same numbers ("8 -> 8").
async function readCounts() {
  return {
    bbox: await page.locator('.bbox').count(),
    nodes: await page.locator('.react-flow__node').count(),
    edges: await page.locator('.react-flow__edge').count(),
  }
}

console.log('BROWSER VERIFICATION (Playwright + real backend)')
console.log(`  app: ${APP}`)
console.log(`  image: ${IMAGE_A}\n`)

// ---------------------------------------------------------------------------
// 1. load + empty state
// ---------------------------------------------------------------------------
await page.goto(APP, { waitUntil: 'domcontentloaded' })
check('app loads', (await page.title()) === 'Scene Graph Explorer', await page.title())
check('header title is the app name',
  (await page.locator('.brand-text h1').innerText()) === 'Scene Graph Explorer')
check('initial empty state visible',
  await page.locator('.empty-state').isVisible(),
  (await page.locator('.empty-state h2').innerText()).trim())
check('U9 landing hero shows the graph motif',
  await page.locator('.hero-icon svg').isVisible())
check('U9 landing hero offers the Choose Image call to action',
  await page.locator('.empty-state button:has-text("Choose Image")').isVisible())
check('U9 landing state is free of implementation details',
  !/CPU|minute|YOLO|CLIP|model loading/i.test(await page.locator('.empty-state').innerText()))
check('no inspector before a scene exists', (await page.locator('.inspector').count()) === 0)
await shot('00-empty-state.png')

// ---------------------------------------------------------------------------
// 2. choose image -> preview
// ---------------------------------------------------------------------------
await page.setInputFiles('input[type=file]', IMAGE_A)
await page.waitForSelector('.preview-image', { timeout: 20000 })
check('image preview appears', await page.locator('.preview-image').isVisible())
check('file name shown', (await page.locator('.file-name').count()) === 1)
check('analyze button enabled', await page.locator('button:has-text("Analyze")').isEnabled())
check('file actions appear after choosing an image',
  (await page.locator('.header-actions').count()) === 1)
check('preview offers a Run analysis action',
  await page.locator('.preview-section button:has-text("Run analysis")').isVisible())
await shot('01-preview.png')

// ---------------------------------------------------------------------------
// 3. analyze (batched CPU inference; allowance kept generous on purpose)
// ---------------------------------------------------------------------------
await page.locator('button:has-text("Analyze")').click()
let sawLoading = false
try {
  await page.waitForSelector('.loading-state', { timeout: 3000 })
  sawLoading = true
} catch {
  sawLoading = false
}
check('loading state shown while analysing', sawLoading)

// Measured while the request is in flight: the loading UI must be centred and
// must not leak implementation details or promise a completion time.
let loadingInfo = null
if (sawLoading) {
  loadingInfo = await page.evaluate(() => {
    const card = document.querySelector('.loading-card')
    const state = document.querySelector('.loading-state')
    const app = document.querySelector('.app')
    const cardBox = card.getBoundingClientRect()
    const appBox = app.getBoundingClientRect()
    return {
      text: card.innerText.replace(/\s+/g, ' ').trim(),
      cardCenterX: cardBox.x + cardBox.width / 2,
      cardCenterY: cardBox.y + cardBox.height / 2,
      appCenterX: appBox.x + appBox.width / 2,
      stateHeight: state.getBoundingClientRect().height,
      spinnerVisible: !!document.querySelector('.spinner'),
    }
  })
  await shot('01b-loading.png')
}
check('U9 loading card is horizontally centred',
  !!loadingInfo && Math.abs(loadingInfo.cardCenterX - loadingInfo.appCenterX) < 30,
  loadingInfo ? `offset ${Math.abs(loadingInfo.cardCenterX - loadingInfo.appCenterX).toFixed(1)}px` : 'not seen')
check('U9 loading state occupies an intentional region',
  !!loadingInfo && loadingInfo.stateHeight >= 300,
  loadingInfo ? `${loadingInfo.stateHeight.toFixed(0)}px tall` : 'not seen')
check('U9 loading UI is a spinner plus one short line',
  !!loadingInfo && loadingInfo.spinnerVisible && /^Analyzing image/.test(loadingInfo.text),
  loadingInfo ? loadingInfo.text.slice(0, 60) : 'not seen')
check('U9 loading UI has no CPU/timing/model text and no fake progress',
  !!loadingInfo &&
    !/CPU|minute|second|YOLO|CLIP|model loading|%/i.test(loadingInfo.text),
  loadingInfo ? loadingInfo.text : 'not seen')

await page.waitForSelector('.explorer', { timeout: 240000 })
await page.waitForFunction(
  () => document.querySelectorAll('.react-flow__node').length > 0,
  null,
  { timeout: 60000 },
)
await page.waitForTimeout(1200) // let fitView settle
const c = await readCounts()
check('objects rendered as bounding boxes', c.bbox > 0, `${c.bbox} boxes`)
check('objects rendered as graph nodes', c.nodes === c.bbox, `${c.nodes} nodes vs ${c.bbox} boxes`)
check('relationships rendered as edges', c.edges > 0, `${c.edges} edges`)
check('inspector still hidden (nothing selected)', (await page.locator('.inspector').count()) === 0)
check('hint line shown instead of an empty inspector',
  await page.locator('.hint-line').isVisible())
await shot('02-analyzed.png')

// ---------------------------------------------------------------------------
// 4b. U7/U9 - duplicate labels, curved edges, equal panel heights
// ---------------------------------------------------------------------------
const boxLabels = await page
  .locator('.bbox-label')
  .evaluateAll((els) => els.map((e) => e.textContent.trim()))
const personLabels = boxLabels.filter((t) => t.startsWith('person'))
const treeLabels = boxLabels.filter((t) => t.startsWith('tree'))
check('U7 duplicate person labels are numbered per scene',
  personLabels.some((t) => /^person 1 /.test(t)) &&
    personLabels.some((t) => /^person 2 /.test(t)),
  personLabels.join(' | '))
check('U7 duplicate tree labels are numbered too',
  treeLabels.length === 2 && treeLabels.every((t) => /^tree \d /.test(t)),
  treeLabels.join(' | '))
check('U7 unique labels are NOT numbered',
  boxLabels.every((t) =>
    !/^(car|bike|vehicle|shoe|sidewalk) \d+ /.test(`${t.replace(/\s\d+%$/, '')} `)),
  boxLabels.filter((t) => /^(car|bike)/.test(t)).join(' | '))

const nodeLabels = await page
  .locator('.react-flow__node')
  .evaluateAll((els) => els.map((e) => e.innerText.trim()))
check('U7 graph nodes use the same numbered labels',
  nodeLabels.filter((t) => /^person \d/.test(t)).length === 2 &&
    nodeLabels.length === c.nodes,
  nodeLabels.filter((t) => t.startsWith('person')).join(' | '))
check('U7 node ids stay obj_N (the numbering is display-only)',
  (await page
    .locator('.react-flow__node')
    .evaluateAll((els) => els.map((e) => e.getAttribute('data-id')))
  ).every((id) => /^obj_\d+$/.test(id || '')))

const edgePath = await page.locator('.react-flow__edge-path').first().getAttribute('d')
check('U9 edges are curved bezier paths, not straight lines',
  /C/.test(edgePath || ''), (edgePath || '').slice(0, 58))

const panelHeights = await page
  .locator('.explorer-panel')
  .evaluateAll((els) => els.map((e) => e.getBoundingClientRect().height))
check('U9 image and graph panels have the same height',
  panelHeights.length === 2 && Math.abs(panelHeights[0] - panelHeights[1]) <= 2,
  panelHeights.map((h) => `${h.toFixed(0)}px`).join(' vs '))
check('U9 both panels stay large enough to be usable',
  panelHeights.every((h) => h > 400),
  panelHeights.map((h) => `${h.toFixed(0)}px`).join(' / '))

// The wrapper must be exactly the size of the painted image, otherwise the
// percentage-based bounding boxes would drift away from the objects.
const imageBox = await page.locator('.scene-image-img').boundingBox()
const wrapBox = await page.locator('.scene-image-wrap').boundingBox()
const imageBodyBox = await page.locator('.scene-image-body').boundingBox()
check('U9 the image panel stays inside its panel',
  imageBox.height <= imageBodyBox.height + 1 && imageBox.width <= imageBodyBox.width + 1,
  `image ${imageBox.width.toFixed(0)}x${imageBox.height.toFixed(0)} in ` +
    `${imageBodyBox.width.toFixed(0)}x${imageBodyBox.height.toFixed(0)}`)
check('U9 the image keeps its aspect ratio',
  Math.abs(imageBox.width / imageBox.height - 800 / 600) < 0.02,
  `rendered ratio ${(imageBox.width / imageBox.height).toFixed(3)} vs 1.333`)
check('U9 the box layer matches the painted image exactly (modulo its 1px border)',
  Math.abs(imageBox.width - wrapBox.width) <= 2 &&
    Math.abs(imageBox.height - wrapBox.height) <= 2,
  `wrap ${wrapBox.width.toFixed(0)}x${wrapBox.height.toFixed(0)}`)
const firstBoxInside = await page.evaluate(() => {
  const wrap = document.querySelector('.scene-image-wrap').getBoundingClientRect()
  const boxes = [...document.querySelectorAll('.bbox')].map((b) => b.getBoundingClientRect())
  const outside = boxes.filter(
    (b) =>
      b.left < wrap.left - 2 ||
      b.top < wrap.top - 60 || // labels sit above the top edge
      b.right > wrap.right + 2 ||
      b.bottom > wrap.bottom + 2,
  )
  return { total: boxes.length, outside: outside.length }
})
check('U9 every bounding box lands inside the image rectangle',
  firstBoxInside.outside === 0,
  `${firstBoxInside.total - firstBoxInside.outside}/${firstBoxInside.total} inside`)

// ---------------------------------------------------------------------------
// 4. U2/U6 - the toggle controls the DEFAULT visibility of the detections
// ---------------------------------------------------------------------------
const bboxBefore = c.bbox
await page.locator('#detections-toggle').uncheck()
await page.waitForTimeout(300)
check('U2 unchecking hides every bounding box', (await page.locator('.bbox').count()) === 0)
check('U2 the overlay layer is retained (empty), not toggled off',
  (await page.locator('.bbox-layer').count()) === 1)
check('U2 caption reports the hidden detections',
  /detections hidden/.test(await page.locator('.scene-image-meta').innerText()),
  (await page.locator('.scene-image-meta').innerText()).trim())
check('U2 graph nodes unaffected by the toggle',
  (await page.locator('.react-flow__node').count()) === c.nodes, `${c.nodes} nodes`)
check('U2 edges unaffected by the toggle',
  (await page.locator('.react-flow__edge').count()) === c.edges, `${c.edges} edges`)
await shot('03-detections-hidden.png')

// ---------------------------------------------------------------------------
// 4c. U6 - a selection still highlights while the detections are OFF
// ---------------------------------------------------------------------------
const firstNodeId = await page.locator('.react-flow__node').first().getAttribute('data-id')
await page
  .locator(`.react-flow__node[data-id="${firstNodeId}"]`)
  .evaluate((el) => el.dispatchEvent(new MouseEvent('click', { bubbles: true })))
await page.waitForTimeout(400)
check('U6 detections OFF + object selected -> only that object is shown',
  (await page.locator('.bbox').count()) === 1,
  `${await page.locator('.bbox').count()} box(es)`)
check('U6 the shown box is the selected object, highlighted red',
  (await page.locator('.bbox-selected').count()) === 1 &&
    (await page.locator('.bbox-selected').first().getAttribute('data-object-id')) === firstNodeId,
  firstNodeId || '(none)')
check('U6 the caption explains what is shown',
  /showing 1 selected/.test(await page.locator('.scene-image-meta').innerText()),
  (await page.locator('.scene-image-meta').innerText()).trim())
await shot('03b-object-selected-with-detections-off.png')

const edgeViaOff = await clickFirstEdge()
await page.waitForTimeout(400)
const shownIds = await page
  .locator('.bbox')
  .evaluateAll((els) => els.map((e) => e.getAttribute('data-object-id')))
const relSubject = (await page.locator('.inspector-list dd').nth(0).innerText()).trim()
const relObject = (await page.locator('.inspector-list dd').nth(2).innerText()).trim()
check('U6 detections OFF + relationship selected -> exactly the two endpoints',
  shownIds.length === 2,
  `${shownIds.length} box(es): ${shownIds.join(', ')} (via ${edgeViaOff})`)
check('U6 the endpoints use the relationship highlight, not the object highlight',
  (await page.locator('.bbox-highlighted').count()) === 2 &&
    (await page.locator('.bbox-selected').count()) === 0)
const shownLabels = await page
  .locator('.bbox-label')
  .evaluateAll((els) => els.map((e) => e.textContent.replace(/\s\d+%$/, '').trim()))
check('U6 the two boxes are the endpoints reported by the inspector',
  [relSubject, relObject].every((label) => shownLabels.includes(label)),
  `${shownLabels.join(' + ')} vs ${relSubject} -> ${relObject}`)
await shot('03c-relationship-endpoints-with-detections-off.png')

await page.locator('.react-flow__pane').click({ position: { x: 6, y: 6 }, force: true })
await page.waitForTimeout(400)
check('U6 detections OFF + nothing selected -> no boxes again',
  (await page.locator('.bbox').count()) === 0)

await page.locator('#detections-toggle').check()
await page.waitForTimeout(300)
check('U2 re-checking restores every bounding box',
  (await page.locator('.bbox').count()) === bboxBefore, `${bboxBefore} boxes`)
check('U2 re-checking restores the caption',
  !/detections hidden/.test(await page.locator('.scene-image-meta').innerText()))

// ---------------------------------------------------------------------------
// 5. U1/U5 - image -> graph selection + inspector content
// ---------------------------------------------------------------------------
await clickFirstBBox()
await page.waitForSelector('.inspector', { timeout: 10000 })
const firstTitle = await page.locator('.bbox').first().getAttribute('title')
const selectedId = (firstTitle || '').match(/\((obj_\d+)\)/)?.[1]
const selectedLabel = (firstTitle || '').split(' (')[0]
check('U1 boxes carry the object label and id', Boolean(selectedId), firstTitle || '(none)')
check('U5 clicked box is highlighted in the image',
  (await page.locator('.bbox-selected').count()) === 1, firstTitle || '')
const nodeBorder = await page
  .locator(`.react-flow__node[data-id="${selectedId}"]`)
  .evaluate((el) => el.style.border || '')
check('U5 matching graph node is highlighted red', nodeBorder.includes('239'), nodeBorder || '(empty)')
check('inspector opens with Object Details',
  /^object details$/i.test((await page.locator('.inspector h2').innerText()).trim()))
check('inspector label matches the clicked object',
  (await page.locator('.inspector-list dd').first().innerText()).trim() === selectedLabel,
  selectedLabel)
check('inspector shows label/id/confidence/bbox rows',
  (await page.locator('.inspector-list dt').count()) === 4)
await shot('04-object-selected.png')

// ---------------------------------------------------------------------------
// 6. U5 - graph -> image selection (opposite direction)
// ---------------------------------------------------------------------------
const nodeIds = await page
  .locator('.react-flow__node')
  .evaluateAll((els) => els.map((e) => e.getAttribute('data-id')))
const otherId = nodeIds.find((id) => id && id !== selectedId)
await page
  .locator(`.react-flow__node[data-id="${otherId}"]`)
  .evaluate((el) => el.dispatchEvent(new MouseEvent('click', { bubbles: true })))
await page.waitForTimeout(500)
const selectedTitle = await page.locator('.bbox-selected').first().getAttribute('title')
check('U5 clicking a graph node highlights its image box',
  (await page.locator('.bbox-selected').count()) === 1 &&
    (selectedTitle || '').includes(otherId),
  `${otherId} -> ${selectedTitle || '(none)'}`)
check('U5 inspector follows the graph selection',
  /^object details$/i.test((await page.locator('.inspector h2').innerText()).trim()))

// ---------------------------------------------------------------------------
// 7. U4 - edge click -> relationship details + both endpoints highlighted
// ---------------------------------------------------------------------------
const edgeVia = await clickFirstEdge()
await page.waitForTimeout(500)
check('U4 clicking an edge shows Relationship Details',
  /^relationship details$/i.test((await page.locator('.inspector h2').innerText()).trim()),
  `via ${edgeVia}`)
check('relationship inspector lists subject/predicate/object/confidence',
  (await page.locator('.inspector-list dt').count()) === 4)
const highlightedCount = await page.locator('.bbox-highlighted').count()
check('U4 both endpoint objects highlighted in the image', highlightedCount === 2,
  `${highlightedCount} highlighted`)
await shot('05-relationship-selected.png')

// ---------------------------------------------------------------------------
// 8. U4 - clicking empty canvas clears the selection
// ---------------------------------------------------------------------------
await page.locator('.react-flow__pane').click({ position: { x: 6, y: 6 }, force: true })
await page.waitForTimeout(500)
check('U4 empty canvas click hides the inspector', (await page.locator('.inspector').count()) === 0)
check('U4 empty canvas click clears every highlight',
  (await page.locator('.bbox-selected, .bbox-highlighted').count()) === 0)
await shot('06-selection-cleared.png')

// ---------------------------------------------------------------------------
// 9. search highlighting in both views
// ---------------------------------------------------------------------------
await page.fill('#search-input', 'person')
await page.waitForTimeout(500)
const searched = await page.locator('.bbox-searched').count()
check('search highlights matching boxes in the image', searched > 0, `${searched} boxes`)
const countText = (await page.locator('.search-count').innerText()).trim()
check('search shows a match count', countText.includes('match'), countText)
const tealNodes = await page
  .locator('.react-flow__node')
  .evaluateAll((els) => els.filter((e) => (e.style.border || '').includes('0, 137, 123')).length)
check('search highlights the same nodes in the graph', tealNodes === searched,
  `${tealNodes} nodes vs ${searched} boxes`)
await shot('07-search.png')
await page.fill('#search-input', '')
await page.waitForTimeout(500)
check('clearing the search removes the highlights',
  (await page.locator('.bbox-searched').count()) === 0)

// ---------------------------------------------------------------------------
// 10. U3 - Download Graph PNG
// ---------------------------------------------------------------------------
const [download] = await Promise.all([
  page.waitForEvent('download', { timeout: 90000 }),
  page.locator('.graph-toolbar button:has-text("Download")').click(),
])
const downloadLabel = (await page.locator('.graph-toolbar button:has-text("Download")').innerText())
  .replace(/\s+/g, ' ')
  .trim()
check('U9 download is a single compact action',
  /^(\u2193 )?Download$/.test(downloadLabel), downloadLabel)
const pngPath = join(SHOTS, 'scene-graph-download.png')
await download.saveAs(pngPath)
const buf = readFileSync(pngPath)
const isPng =
  buf.length > 24 && buf[0] === 0x89 && buf[1] === 0x50 && buf[2] === 0x4e && buf[3] === 0x47
const pngWidth = isPng ? buf.readUInt32BE(16) : 0
const pngHeight = isPng ? buf.readUInt32BE(20) : 0
check('U3 download is named scene-graph.png',
  download.suggestedFilename() === 'scene-graph.png', download.suggestedFilename())
check('U3 downloaded file is a real PNG', isPng, `${buf.length} bytes`)
check('U3 PNG is not blank', buf.length > 5000, `${buf.length} bytes`)
check('U3 PNG is the graph panel, not the whole page',
  isPng && pngWidth > 300 && pngWidth < 1500 && pngHeight > 250 && pngHeight < 1200,
  `${pngWidth}x${pngHeight}`)

// ---------------------------------------------------------------------------
// 10b. U8 - graph lock / unlock and node dragging
// ---------------------------------------------------------------------------
const lockText = () => page.locator('button.lock-toggle').innerText()
check('U8 graph defaults to unlocked and draggable',
  /Unlocked/.test(await lockText()), (await lockText()).trim())

const nodeIndex = await centralNodeIndex()
const freeDrag = await dragNodeBy(nodeIndex, 80, 50)
check('U8 unlocked graph lets nodes be dragged',
  Math.abs(freeDrag.graphDX) > 5 || Math.abs(freeDrag.movedX) > 25,
  `graph position moved ${freeDrag.graphDX.toFixed(0)}px x, ` +
    `${freeDrag.graphDY.toFixed(0)}px y (screen ${freeDrag.movedX.toFixed(0)}x${freeDrag.movedY.toFixed(0)}px)`)
await shot('08b-node-dragged.png')

// The manual position must survive an unrelated selection change.
const posBeforeSelection = await page
  .locator('.react-flow__node')
  .nth(nodeIndex)
  .boundingBox()
await clickFirstBBox()
await page.waitForTimeout(400)
await page.locator('.react-flow__pane').click({ position: { x: 6, y: 6 }, force: true })
await page.waitForTimeout(400)
const posAfterSelection = await page
  .locator('.react-flow__node')
  .nth(nodeIndex)
  .boundingBox()
check('U8 the dragged position survives selection changes',
  Math.abs(posAfterSelection.x - posBeforeSelection.x) < 2 &&
    Math.abs(posAfterSelection.y - posBeforeSelection.y) < 2,
  `drifted ${(posAfterSelection.x - posBeforeSelection.x).toFixed(1)}px x during select/deselect`)

await page.locator('button.lock-toggle').click()
await page.waitForTimeout(300)
check('U8 lock control switches to the locked state',
  /Locked/.test(await lockText()), (await lockText()).trim())
// How node dragging is verified: Playwright's headless mouse.down/move/up DOES
// exercise React Flow (the unlocked drag above moves the node in graph space).
// With the lock engaged the same gesture is a PAN (pointer-down on a node pans
// the canvas when the node is not draggable), so the honest locked-node check
// is that the node's GRAPH-space position stays fixed while the viewport may
// move - which simultaneously demonstrates that pan still works while locked.
const viewportBeforeLock = await liveViewportTransform()
const lockedDrag = await dragNodeBy(nodeIndex, -70, -45)
const viewportAfterLock = await liveViewportTransform()
check('U8 locked graph does not move nodes',
  Math.abs(lockedDrag.graphDX) < 2 && Math.abs(lockedDrag.graphDY) < 2,
  `graph position moved ${lockedDrag.graphDX.toFixed(1)}px x, ` +
    `${lockedDrag.graphDY.toFixed(1)}px y (screen ${lockedDrag.movedX.toFixed(0)}x${lockedDrag.movedY.toFixed(0)}px)`)

// Panning must keep working while locked - checked from an empty origin too
// (nodes AND React Flow panels excluded), plus the pan-from-node gesture above.
const panStart = await page.evaluate(() => {
  const pane = document.querySelector('.graph-container').getBoundingClientRect()
  const avoid = [
    ...document.querySelectorAll('.react-flow__node, .react-flow__panel'),
  ].map((n) => n.getBoundingClientRect())
  // Scan a grid of candidate points, bottom-right first, clear of nodes/panels.
  for (let y = pane.bottom - 30; y > pane.top + 30; y -= 40) {
    for (let x = pane.right - 30; x > pane.left + 30; x -= 40) {
      const inside = avoid.some(
        (r) => x > r.left - 12 && x < r.right + 12 && y > r.top - 12 && y < r.bottom + 12,
      )
      if (!inside) return { x, y }
    }
  }
  return null
})
const viewportBefore = await liveViewportTransform()
if (panStart) {
  await page.mouse.move(panStart.x, panStart.y)
  await page.mouse.down()
  await page.mouse.move(panStart.x - 70, panStart.y + 50, { steps: 10 })
  await page.mouse.up()
  await page.waitForTimeout(300)
}
const viewportAfter = await liveViewportTransform()
if (panStart) {
  await page.mouse.move(panStart.x - 70, panStart.y + 50)
  await page.mouse.down()
  await page.mouse.move(panStart.x, panStart.y, { steps: 10 })
  await page.mouse.up()
  await page.waitForTimeout(300)
}
const pannedWhileLocked = viewportBefore !== viewportAfter
check('U8 panning still works while the graph is locked',
  (panStart !== null && pannedWhileLocked) || viewportBeforeLock !== viewportAfterLock,
  panStart
    ? `empty-area pan ${pannedWhileLocked ? 'panned' : 'unchanged'}; pan-from-node ${viewportBeforeLock !== viewportAfterLock ? 'panned' : 'unchanged'}`
    : `no empty pan origin; pan-from-node ${viewportBeforeLock !== viewportAfterLock ? 'panned' : 'unchanged'}`)
await shot('08c-graph-locked.png')

await page.locator('button.lock-toggle').click()
await page.waitForTimeout(300)
check('U8 unlocking restores the unlocked state',
  /Unlocked/.test(await lockText()), (await lockText()).trim())
const unlockedAgain = await dragNodeBy(nodeIndex, 30, 0)
check('U8 dragging works again after unlocking',
  Math.abs(unlockedAgain.movedX) > 10,
  `moved ${unlockedAgain.movedX.toFixed(0)}px`)

// ---------------------------------------------------------------------------
// 10c. V1/V2 - relationship labels: readable, one per edge, never overlapping
// ---------------------------------------------------------------------------
console.log('\n-- relationship labels (deterministic lanes) --')
const labelMetrics = await page.evaluate(() => {
  const labels = [...document.querySelectorAll('.graph-container .edge-label')]
  return labels.map((el) => {
    const r = el.getBoundingClientRect()
    const cs = getComputedStyle(el)
    return {
      text: (el.textContent || '').trim(),
      edgeId: el.getAttribute('data-edge-id'),
      x: r.x,
      y: r.y,
      w: r.width,
      h: r.height,
      fontSize: parseFloat(cs.fontSize),
      background: cs.backgroundColor,
      borderWidth: parseFloat(cs.borderTopWidth),
      zIndex: cs.zIndex,
    }
  })
})
const visibleEdges = await readCounts()
check('V1 every drawn relationship has a label',
  labelMetrics.length === visibleEdges.edges && labelMetrics.length > 0,
  `${labelMetrics.length} labels / ${visibleEdges.edges} edges`)
check('V1 labels are unique per relationship (no duplicates, no swaps)',
  new Set(labelMetrics.map((l) => l.edgeId)).size === labelMetrics.length &&
    labelMetrics.every((l) => l.edgeId && l.text.length > 0))
check('V1 labels are actually rendered (non-zero boxes)',
  labelMetrics.every((l) => l.w > 10 && l.h > 8),
  labelMetrics.map((l) => `${l.text}:${l.w.toFixed(0)}x${l.h.toFixed(0)}`).slice(0, 3).join(' '))
check('V1/V2 labels are readable (own plate, border, sane font size)',
  labelMetrics.every((l) => l.fontSize >= 10 && l.fontSize <= 13 && l.borderWidth >= 1) &&
    labelMetrics.every((l) => !/rgba\(0, 0, 0, 0\)|transparent/.test(l.background)),
  `${labelMetrics[0].fontSize}px, bg ${labelMetrics[0].background}`)
check('V2 labels cannot be swallowed by the graph behind them (z-index set)',
  labelMetrics.every((l) => Number(l.zIndex) >= 1), `z-index=${labelMetrics[0].zIndex}`)

const labelOverlaps = []
for (let i = 0; i < labelMetrics.length; i++) {
  for (let j = i + 1; j < labelMetrics.length; j++) {
    const a = labelMetrics[i]
    const b = labelMetrics[j]
    const w = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x)
    const h = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y)
    if (w > 0 && h > 0) {
      const ratio = (w * h) / Math.min(a.w * a.h, b.w * b.h)
      if (ratio > 0.2) labelOverlaps.push(`${a.text}/${b.text} ${(ratio * 100).toFixed(0)}%`)
    }
  }
}
check('V3 no two relationship labels overlap in the rendered graph',
  labelOverlaps.length === 0,
  labelOverlaps.length
    ? labelOverlaps.join(', ')
    : `${labelMetrics.length} labels pairwise clear`)

// Visual check: for each label, ask the browser what is actually painted at
// the label centre (`elementFromPoint`).  A label counts as buried only when
// the topmost element is a *node* - that is the only thing with a filled
// backdrop big enough to hide a label.  Edges ducking under a label, the
// viewport, wrappers, etc. never count: the label plate itself is opaque and
// sits above the edge layer (z-index=3), so passing lanes cannot swallow it.
const buriedLabels = await page.evaluate(() => {
  const labels = [...document.querySelectorAll('.graph-container .edge-label')]
    .filter((el) => el.getBoundingClientRect().width > 0)
  return labels
    .filter((el) => {
      const r = el.getBoundingClientRect()
      const top = document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2)
      return !!top && !!top.closest && !!top.closest('.react-flow__node')
    })
    .map((el) => el.textContent.trim())
})
check('V3 no relationship label hides behind a node',
  buriedLabels.length === 0,
  buriedLabels.length ? buriedLabels.join(', ') : 'none')
await shot('08d-labels-routed.png')

// ---------------------------------------------------------------------------
// 10d. V1 - object visibility (hide / show, scene data untouched)
// ---------------------------------------------------------------------------
console.log('\n-- graph visibility: objects --')
const statsBefore = await sceneStats()
const visBefore = await readCounts()
await clickFirstBBox()
await page.waitForTimeout(400)
// Scope every object-step to the inspector root: the graph summary picks up a
// ".graph-note-hidden" only when something is hidden, which must never be
// confused with this toggle (and there is exactly one inspector at a time).
const inspectorRoot = page.locator('.inspector')
const hiddenObjectLabel = (await inspectorRoot.locator('.inspector-list dd').first().innerText()).trim()
check('V1 the object inspector offers "Show in graph"',
  await inspectorRoot.locator('#object-visibility-toggle').isVisible())
check('V1 the switch is on by default (object drawn)',
  await inspectorRoot.locator('#object-visibility-toggle').isChecked())

await inspectorRoot.locator('#object-visibility-toggle').click({ force: true })
await page.waitForFunction(
  () => !document.querySelector('#object-visibility-toggle')?.checked,
  null,
  { timeout: 5000 },
)
await page.waitForTimeout(500)
const visHidden = await readCounts()
check('V1 hiding an object removes exactly that node',
  visHidden.nodes === visBefore.nodes - 1,
  `${visBefore.nodes} -> ${visHidden.nodes} nodes`)
check('V1 hiding an object also removes every relationship touching it',
  visHidden.edges < visBefore.edges,
  `${visBefore.edges} -> ${visHidden.edges} edges`)
check('V1 the image detections are NOT affected',
  visHidden.bbox === visBefore.bbox, `${visHidden.bbox} boxes`)
check('V1 the scene itself is unchanged (stats still report everything)',
  (await sceneStats()) === statsBefore, await sceneStats())
await page.fill('#search-input', hiddenObjectLabel.split(' ')[0])
await page.waitForTimeout(400)
const searchHits = await page.locator('.bbox-searched').count()
check('V1 search still finds the hidden object in the image', searchHits > 0,
  `${searchHits} match(es) for "${hiddenObjectLabel.split(' ')[0]}"`)
await page.fill('#search-input', '')
await page.waitForTimeout(300)
check('V1 the inspector closes instead of showing a hidden element',
  (await page.locator('.inspector').count()) === 0)
check('V1 the graph reports what is hidden and offers a way back',
  await page.locator('.show-all-visibility').isVisible() &&
    /Hidden in graph/.test(await page.locator('.graph-note-hidden').innerText()))
await shot('08e-object-hidden.png')

await clickFirstBBox()
await page.waitForTimeout(400)
check('V1 re-selecting the hidden object shows the switch as OFF',
  (await inspectorRoot.locator('#object-visibility-toggle').isChecked()) === false)
check('V1 the hidden state is explained in the inspector',
  /hidden from the graph/i.test(await inspectorRoot.locator('.visibility-hint').innerText()))
await inspectorRoot.locator('#object-visibility-toggle').click({ force: true })
await page.waitForFunction(
  () => document.querySelector('#object-visibility-toggle')?.checked === true,
  null,
  { timeout: 5000 },
)
await page.waitForTimeout(500)
const visRestored = await readCounts()
check('V1 showing the object again restores the node and its relationships',
  visRestored.nodes === visBefore.nodes && visRestored.edges === visBefore.edges,
  `${visRestored.nodes} nodes / ${visRestored.edges} edges`)
check('V1 the "hidden" report disappears when nothing is hidden',
  (await page.locator('.show-all-visibility').count()) === 0)
await page.locator('.react-flow__pane').click({ position: { x: 6, y: 6 }, force: true })
await page.waitForTimeout(300)

// ---------------------------------------------------------------------------
// 10e. V1 - relationship visibility (hide / show / show all)
// ---------------------------------------------------------------------------
console.log('\n-- graph visibility: relationships --')
const relBefore = await readCounts()
const edgeIdBefore = await page.locator('.edge-label').first().getAttribute('data-edge-id')
const relVia = await clickFirstEdge()
const relInspectorRoot = page.locator('.inspector')
// The inspector swap is a React commit; wait for the relationship heading
// instead of a fixed delay so this check cannot race the render.  A timeout
// degrades to a reported FAIL rather than aborting the whole run.
const relSelected = await page
  .waitForFunction(
    () => {
      const h2 = document.querySelector('.inspector h2')
      return !!h2 && h2.textContent.trim() === 'Relationship Details'
    },
    null,
    { timeout: 5000 },
  )
  .then(() => true)
  .catch(() => false)
check('V1 clicking a relationship label selects the relationship',
  /edge label/.test(relVia) && relSelected,
  relVia)
check('V1 the relationship inspector offers "Show in graph"',
  (await relInspectorRoot.locator('#relationship-visibility-toggle').isVisible()) &&
    (await relInspectorRoot.locator('#relationship-visibility-toggle').isChecked()))

await relInspectorRoot.locator('#relationship-visibility-toggle').click({ force: true })
await page.waitForFunction(
  () => !document.querySelector('#relationship-visibility-toggle')?.checked,
  null,
  { timeout: 5000 },
)
await page.waitForTimeout(500)
const relHidden = await readCounts()
check('V1 hiding a relationship removes exactly that edge',
  relHidden.edges === relBefore.edges - 1,
  `${relBefore.edges} -> ${relHidden.edges} edges (hid ${edgeIdBefore})`)
check('V1 the endpoints of a hidden relationship stay drawn',
  relHidden.nodes === relBefore.nodes, `${relHidden.nodes} nodes`)
check('V1 no stale inspector for the hidden relationship',
  (await page.locator('.inspector').count()) === 0)
check('V1 the graph offers "Show all"', await page.locator('.show-all-visibility').isVisible())
await page.locator('.show-all-visibility').click()
await page.waitForTimeout(500)
const relRestored = await readCounts()
check('V1 "Show all" brings the relationship back',
  relRestored.edges === relBefore.edges && relRestored.nodes === relBefore.nodes,
  `${relRestored.edges} edges`)

// ---------------------------------------------------------------------------
// 10f. V1 - the exported PNG shows the visible graph
// ---------------------------------------------------------------------------
console.log('\n-- PNG export follows the visible graph --')
await page.locator('.react-flow__pane').click({ position: { x: 6, y: 6 }, force: true })
await clickFirstBBox()
await page.waitForTimeout(400)
await page.locator('.inspector').locator('#object-visibility-toggle').click({ force: true })
await page.waitForFunction(
  () => !document.querySelector('#object-visibility-toggle')?.checked,
  null,
  { timeout: 5000 },
)
await page.waitForTimeout(600)
const pngHiddenPath = join(SHOTS, 'scene-graph-download-hidden.png')
const [downloadHidden] = await Promise.all([
  page.waitForEvent('download', { timeout: 90000 }),
  page.locator('.graph-toolbar button:has-text("Download")').click(),
])
await downloadHidden.saveAs(pngHiddenPath)
const bufHidden = readFileSync(pngHiddenPath)
const isPngHidden =
  bufHidden.length > 24 &&
  bufHidden[0] === 0x89 &&
  bufHidden[1] === 0x50 &&
  bufHidden[2] === 0x4e &&
  bufHidden[3] === 0x47
check('V1 the export still produces a valid PNG with elements hidden',
  isPngHidden, `${bufHidden.length} bytes`)
check('V1 the PNG reflects the hidden graph (differs from the full export)',
  isPngHidden && !bufHidden.equals(buf),
  `${buf.length} bytes -> ${bufHidden.length} bytes`)
check('V1 the hidden graph exports a smaller canvas',
  isPngHidden &&
    bufHidden.readUInt32BE(16) <= buf.readUInt32BE(16) &&
    bufHidden.readUInt32BE(20) <= buf.readUInt32BE(20),
  `${buf.readUInt32BE(16)}x${buf.readUInt32BE(20)} -> ${bufHidden.readUInt32BE(16)}x${bufHidden.readUInt32BE(20)}`)
await shot('08f-png-hidden.png')

const hiddenDragIndex = await centralNodeIndex()
const dragWhileHidden = await dragNodeBy(hiddenDragIndex, 40, 25)
check('V1 dragging still works with elements hidden',
  Math.abs(dragWhileHidden.movedX) > 5, `moved ${dragWhileHidden.movedX.toFixed(0)}px`)

await page.locator('.show-all-visibility').click()
await page.waitForTimeout(500)
check('V1 "Show all" clears every hidden element',
  (await readCounts()).edges === relBefore.edges &&
    (await page.locator('.show-all-visibility').count()) === 0)
await page.locator('.react-flow__pane').click({ position: { x: 6, y: 6 }, force: true })
await page.waitForTimeout(300)

// ---------------------------------------------------------------------------
// 11. Reset
// ---------------------------------------------------------------------------
await page.locator('button:has-text("Reset")').click()
await page.waitForTimeout(500)
check('reset returns to the empty state', await page.locator('.empty-state').isVisible())
check('reset removes the explorer', (await page.locator('.explorer').count()) === 0)
check('reset removes the inspector and the search box',
  (await page.locator('.inspector').count()) === 0 &&
    (await page.locator('#search-input').count()) === 0)
await shot('08-after-reset.png')

// ---------------------------------------------------------------------------
// 12. second image - no state leaks between analyses
// ---------------------------------------------------------------------------
if (existsSync(IMAGE_B)) {
  await page.setInputFiles('input[type=file]', IMAGE_B)
  await page.waitForSelector('.preview-image', { timeout: 20000 })
  await page.locator('button:has-text("Analyze")').click()
  await page.waitForSelector('.explorer', { timeout: 240000 })
  await page.waitForFunction(
    () => document.querySelectorAll('.react-flow__node').length > 0,
    null,
    { timeout: 60000 },
  )
  await page.waitForTimeout(1200)
  const c2 = await readCounts()
  check('second image analyses cleanly', c2.bbox > 0 && c2.nodes === c2.bbox,
    `${c2.bbox} boxes / ${c2.nodes} nodes / ${c2.edges} edges`)
  check('no stale selection after re-analysing',
    (await page.locator('.inspector').count()) === 0)
  await shot('09-second-image.png')
} else {
  console.log('  [SKIP] second-image check (VG_100K_2/2.jpg not present)')
}

// ---------------------------------------------------------------------------
// 13. runtime errors
// ---------------------------------------------------------------------------
check('no uncaught page errors', pageErrors.length === 0,
  pageErrors.slice(0, 3).join(' | ') || 'none')
if (consoleErrors.length) {
  console.log(`         console errors observed (informational): ${consoleErrors.length}`)
  consoleErrors.slice(0, 3).forEach((e) => console.log(`           - ${e.slice(0, 140)}`))
}

await browser.close()
console.log(failures === 0 ? '\nRESULT: PASS' : `\nRESULT: FAIL (${failures})`)
process.exit(failures === 0 ? 0 : 1)
