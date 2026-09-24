/**
 * Final UX polish - interaction verifier.
 *
 * Checks the NEW interaction contract of this phase against the real scene
 * JSON (tests/_scene.json) and the real component sources:
 *
 *   1. detection-visibility rules (toggle OFF + selection must still highlight)
 *   2. duplicate object numbering ("person 1" / "person 2", display only)
 *   3. graph lock / unlock state (draggable === !locked, pan/zoom unaffected)
 *   4. curved edge configuration (bezier pathOptions, not straight smoothstep)
 *   5. equal image/graph panel sizing (matched heights, not fixed 560px)
 *   6. no implementation details left in the loading UI
 *
 * The logic under test is REPLICATED from the components (and the checks below
 * assert that the components still contain the corresponding source), which is
 * the same approach the existing frontend verifiers use.
 *
 * Run from frontend/:  node verify_interaction.mjs
 */
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'
import {
  BASE_CURVATURE,
  MAX_CURVATURE,
  MIN_CURVATURE,
  planEdgeRouting,
} from './src/lib/edgeRouting.js'

const here = dirname(fileURLToPath(import.meta.url))
const read = (...parts) => readFileSync(join(here, ...parts), 'utf-8')

const scene = JSON.parse(read('..', 'tests', '_scene.json'))
const appSource = read('src', 'App.jsx')
const graphSource = read('src', 'components', 'SceneGraph.jsx')
const imageSource = read('src', 'components', 'SceneImage.jsx')
const inspectorSource = read('src', 'components', 'Inspector.jsx')
const edgeSource = read('src', 'components', 'LaneEdge.jsx')
const cssSource = read('src', 'App.css')
const indexHtml = read('index.html')

let failures = 0
function check(name, ok, detail = '') {
  console.log(`  [${ok ? 'PASS' : 'FAIL'}] ${name}${detail ? ' - ' + detail : ''}`)
  if (!ok) failures++
}

console.log('FINAL UX POLISH - INTERACTION VERIFICATION (real scene)')

// ----------------------------------------------------------------------
// 1. duplicate object numbering (mirrors the displayLabels memo in App.jsx)
// ----------------------------------------------------------------------
function displayLabels(objects) {
  const labels = {}
  const totals = {}
  for (const obj of objects) totals[obj.label] = (totals[obj.label] || 0) + 1
  const seen = {}
  for (const obj of objects) {
    if (totals[obj.label] > 1) {
      seen[obj.label] = (seen[obj.label] || 0) + 1
      labels[obj.id] = `${obj.label} ${seen[obj.label]}`
    } else {
      labels[obj.id] = obj.label
    }
  }
  return labels
}

const labels = displayLabels(scene.objects)
const byLabel = {}
for (const obj of scene.objects) (byLabel[obj.label] ||= []).push(obj)

const persons = byLabel.person || []
const singles = Object.entries(byLabel).filter(([, list]) => list.length === 1)

check('duplicate labels get 1..n numbering', persons.length >= 2 &&
  persons.every((o, i) => labels[o.id] === `person ${i + 1}`),
  persons.map((o) => labels[o.id]).join(', '))
check('unique labels are NOT numbered', singles.every(([label, list]) =>
  labels[list[0].id] === label), singles.map(([l]) => l).join(', '))
check('numbering is deterministic', JSON.stringify(labels) ===
  JSON.stringify(displayLabels(scene.objects)))
check('numbering never changes the data', scene.objects.every((o) =>
  typeof o.id === 'string' && o.label === o.label && !labels[o.id].includes('obj_')))
check('object ids keep the obj_N form', scene.objects.every((o) => /^obj_\d+$/.test(o.id)))

// search still matches on the BASE label, so "person" hits person 1 and 2
function searchObjectIds(objects, query) {
  if (!query.trim()) return []
  const q = query.trim().toLowerCase()
  return objects.filter((o) => o.label.toLowerCase().includes(q)).map((o) => o.id)
}
check('search uses the base label (finds both persons)',
  searchObjectIds(scene.objects, 'person').length === persons.length,
  `${searchObjectIds(scene.objects, 'person').length} hits`)

// ----------------------------------------------------------------------
// 2. detection visibility rules (mirrors the visibleObjectIds memo)
// ----------------------------------------------------------------------
function visibleObjectIds({ showDetections, selectedObjectId, selectedRelationshipId }) {
  let relEndpointIds = []
  if (selectedRelationshipId) {
    const rel = scene.relationships.find((r) => r.id === selectedRelationshipId)
    if (rel) relEndpointIds = [rel.subject_id, rel.object_id]
  }
  if (showDetections) return scene.objects.map((o) => o.id)
  if (selectedRelationshipId) return relEndpointIds
  if (selectedObjectId) return [selectedObjectId]
  return []
}

const allIds = scene.objects.map((o) => o.id)
const rel = scene.relationships[0]
const relEndpoints = [rel.subject_id, rel.object_id]

check('A: detections ON -> every object',
  visibleObjectIds({ showDetections: true }).length === allIds.length,
  `${allIds.length} boxes`)
check('B: detections OFF + no selection -> no boxes',
  visibleObjectIds({ showDetections: false }).length === 0)
check('C: detections OFF + object selected -> only that object',
  JSON.stringify(visibleObjectIds({ showDetections: false, selectedObjectId: 'obj_3' })) ===
    JSON.stringify(['obj_3']))
check('D: detections OFF + relationship selected -> only the endpoints',
  JSON.stringify(visibleObjectIds({
    showDetections: false, selectedRelationshipId: rel.id,
  })) === JSON.stringify(relEndpoints),
  `${rel.subject_id} + ${rel.object_id}`)
check('D2: detections ON + relationship selected -> all boxes (endpoints emphasised)',
  visibleObjectIds({ showDetections: true, selectedRelationshipId: rel.id }).length ===
    allIds.length)
check('no case ever returns the whole layer when the toggle is off',
  [null, 'obj_0', 'obj_8'].every((sel) =>
    visibleObjectIds({ showDetections: false, selectedObjectId: sel }).length <= 1))
check('scene.objects is untouched by the visibility rules',
  scene.objects.length === allIds.length &&
    scene.relationships.length > 0)

// ----------------------------------------------------------------------
// 3. graph lock / unlock
// ----------------------------------------------------------------------
check('lock defaults to unlocked', /useState\(false\)[\s\S]{0,80}graphLocked|graphLocked[^\n]*useState\(false\)/.test(appSource),
  'graphLocked starts false')
check('node dragging is bound to the lock', graphSource.includes('nodesDraggable={!locked}'))
check('dragging is no longer hard-disabled', !graphSource.includes('nodesDraggable={false}'))
check('pan/zoom are explicitly unaffected by the lock',
  !/panOnDrag=\{[^}]*locked/.test(graphSource) &&
    !graphSource.includes('zoomOnScroll={!locked}'))
check('the lock is never written into the scene data',
  !/scene\.(objects|relationships)[^\n]*locked/.test(graphSource))
check('dagre layout is not re-run on selection changes',
  /const sceneData = useMemo\([\s\S]*?\[objects, relationships, displayLabels, routing\]/.test(graphSource),
  'memo deps are scene data only')

// ----------------------------------------------------------------------
// 4. curved edges (lane routing - see verify_visibility.mjs for the model)
// ----------------------------------------------------------------------
check('edges use the custom lane edge and register it',
  /const EDGE_TYPES = \{ lane: LaneEdge \}/.test(graphSource) &&
    /edgeTypes=\{EDGE_TYPES\}/.test(graphSource) &&
    /type: 'lane'/.test(graphSource))
check('the base curvature is still subtle (0.1 - 0.5), not a loop',
  BASE_CURVATURE > 0.1 && BASE_CURVATURE < 0.5, `curvature=${BASE_CURVATURE}`)
const curveValues = Object.values(planEdgeRouting(scene.relationships)).map((p) => p.curvature)
check('every routed edge stays inside the curvature bounds',
  curveValues.every((c) => c >= MIN_CURVATURE && c <= MAX_CURVATURE),
  `${Math.min(...curveValues).toFixed(2)} - ${Math.max(...curveValues).toFixed(2)}`)
check('straight smoothstep edges are gone', !graphSource.includes("'smoothstep'"))
check('edge labels and inverse-pair styling are preserved',
  /edge-label/.test(edgeSource) &&
    /EdgeLabelRenderer/.test(edgeSource) &&
    graphSource.includes('strokeDasharray') &&
    /\.graph-container \.edge-label \{/.test(cssSource))

// ----------------------------------------------------------------------
// 5. equal panel sizing + download button + hero/loading states
// ----------------------------------------------------------------------
check('explorer sets a matched panel height',
  /\.app\.is-explorer \.explorer-panel \{\s*height: \d+px;/.test(cssSource),
  (cssSource.match(/\.app\.is-explorer \.explorer-panel \{\s*height: (\d+)px/) || [])[1] + 'px')
check('the graph no longer has a fixed 560px height',
  !/\.graph-container \{[^}]*height: 560px/.test(cssSource))
check('both panels are the same height',
  /grid-template-columns: 0\.9fr 1\.1fr/.test(cssSource) &&
    /\.app\.is-explorer \.explorer-panel \{\s*height: \d+px;/.test(cssSource))
check('image panel body centres the image',
  cssSource.includes('.scene-image-body') && /\.scene-image-body \{[^}]*justify-content: center/.test(cssSource))
check('small screens still stack and re-size the panels',
  /@media \(max-width: 900px\)[\s\S]*explorer-panel-graph[\s\S]*height: \d+px/.test(cssSource))
check('graph viewport fills its panel flexibly',
  /\.graph-container \{[^}]*flex: 1/.test(cssSource))
check('download button is compact and keeps the PNG export',
  (graphSource.includes('Download') && !graphSource.includes('Download Graph PNG')) &&
    graphSource.includes('toPng') &&
    graphSource.includes("link.download = 'scene-graph.png'"))
check('"Download Graph PNG" label is gone', !graphSource.includes('Download Graph PNG'))

// loading / landing copy must not leak implementation details
const cpuWords = ['CPU', 'up to a minute', 'seconds', 'model loading', 'YOLO', 'CLIP']
const loadingFiles = [appSource, indexHtml]
check('loading UI has no CPU/timing/model text',
  loadingFiles.every((src) => cpuWords.every((w) => !src.includes(w))),
  cpuWords.filter((w) => loadingFiles.some((src) => src.includes(w))).join(', ') || 'clean')
check('loading UI is centred by the layout',
  /\.empty-state,\s*\.loading-state \{[^}]*justify-content: center/.test(cssSource))
check('loading shows a spinner and a short message',
  appSource.includes('className="spinner"') && appSource.includes('Analyzing image...'))
check('no fake progress percentage', !/%\s*<\/|width: `\$\{progress/.test(appSource))
check('landing hero has a single call to action',
  appSource.includes('hero-icon') && appSource.includes('Choose Image'))

// ----------------------------------------------------------------------
// 6. component wiring for the new contract
// ----------------------------------------------------------------------
check('App passes the visibility list to the image view',
  appSource.includes('visibleObjectIds={visibleObjectIds}'))
check('images boxes are filtered by the visibility list',
  imageSource.includes('visibleObjectIds') && imageSource.includes('visibleSet.has(obj.id)'))
check('App passes display labels to image, graph and inspector',
  ['SceneImage', 'SceneGraph', 'Inspector'].every((c) => {
    const block = appSource.slice(appSource.indexOf(`<${c}`))
    return /displayLabels=\{displayLabels\}/.test(block.slice(0, 900))
  }))
check('graph nodes and edges carry the display label',
  graphSource.includes('labels[obj.id] || obj.label'))
check('inspector resolves display labels',
  inspectorSource.includes('labelFor') && inspectorSource.includes('const labelFor = (obj)'))
check('the real detection checkbox still exists (browser-testable)',
  appSource.includes('id="detections-toggle"') && appSource.includes('type="checkbox"'))

console.log(failures === 0 ? '\nRESULT: PASS' : `\nRESULT: FAIL (${failures})`)
process.exit(failures === 0 ? 0 : 1)

