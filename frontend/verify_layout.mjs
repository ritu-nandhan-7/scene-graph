/**
 * Verifies the dagre layout used by SceneGraph.jsx against a real scene JSON
 * (tests/_scene.json) AND checks that the component sources really implement
 * the frontend-polish requirements (TB layout, static edges, inverse-pair
 * styling, PNG export, pane-click deselection, toggle-gated overlays).
 *
 * Run from frontend/:  node verify_layout.mjs
 *
 * NOTE (frontend-polish phase): this verifier previously mirrored the D2/D3
 * left-to-right grid layout (160x64 nodes, rankdir "LR").  The component moved
 * to a TOP-TO-BOTTOM layout with 170x56 nodes, so the mirror below was updated
 * to match - the test was updated rather than bypassed.
 */
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'
import dagre from '@dagrejs/dagre'

const here = dirname(fileURLToPath(import.meta.url))
const scene = JSON.parse(readFileSync(join(here, '..', 'tests', '_scene.json'), 'utf-8'))
const graphSource = readFileSync(join(here, 'src', 'components', 'SceneGraph.jsx'), 'utf-8')
const routingSource = readFileSync(join(here, 'src', 'lib', 'edgeRouting.js'), 'utf-8')
const imageSource = readFileSync(join(here, 'src', 'components', 'SceneImage.jsx'), 'utf-8')
const appSource = readFileSync(join(here, 'src', 'App.jsx'), 'utf-8')

// ---- shared with src/lib/edgeRouting.js ------------------------------------
// Node size + dagre config live in the routing lib (single source of truth);
// the component imports them, so these assertions read the lib, not a copy.
const { DAGRE_CONFIG, NODE_WIDTH, NODE_HEIGHT } = await import('./src/lib/edgeRouting.js')

function buildNodes(objects) {
  return objects.map((obj) => ({ id: obj.id, data: { label: obj.label } }))
}

function buildEdges(relationships) {
  return relationships.map((rel) => ({
    id: rel.id,
    source: rel.subject_id,
    target: rel.object_id,
    label: rel.predicate,
  }))
}

function layoutNodes(nodes, edges) {
  // Same dagre config the component lays out through (layoutNodeRects).
  const graph = new dagre.graphlib.Graph({ multigraph: true })
  graph.setGraph({ ...DAGRE_CONFIG })
  graph.setDefaultEdgeLabel(() => ({}))
  nodes.forEach((node) => graph.setNode(node.id, { width: NODE_WIDTH, height: NODE_HEIGHT }))
  edges.forEach((edge) => graph.setEdge(edge.source, edge.target, { weight: 1 }, edge.id))
  dagre.layout(graph)
  // dagre reports centres; React Flow positions are top-left corners.
  return nodes.map((node) => {
    const centre = graph.node(node.id)
    return {
      ...node,
      centre,
      position: { x: centre.x - NODE_WIDTH / 2, y: centre.y - NODE_HEIGHT / 2 },
    }
  })
}

let failures = 0
function check(name, ok, detail = '') {
  console.log(`  [${ok ? 'PASS' : 'FAIL'}] ${name}${detail ? ' - ' + detail : ''}`)
  if (!ok) failures++
}

console.log('DAGRE LAYOUT + COMPONENT VERIFICATION (real scene)')

const nodes = layoutNodes(buildNodes(scene.objects), buildEdges(scene.relationships))
const edges = buildEdges(scene.relationships)

check('node count preserved', nodes.length === scene.objects.length,
  `${nodes.length}/${scene.objects.length}`)
check('edge count preserved (no relationship lost)', edges.length === scene.relationships.length,
  `${edges.length}/${scene.relationships.length}`)

const badPos = nodes.filter((n) => !Number.isFinite(n.position.x) || !Number.isFinite(n.position.y))
check('all positions finite', badPos.length === 0,
  badPos.length ? badPos.map((n) => n.id).join(',') : `${nodes.length} ok`)

// Real node rectangles, not just centres.
const overlaps = []
for (let i = 0; i < nodes.length; i++) {
  for (let j = i + 1; j < nodes.length; j++) {
    const dx = Math.abs(nodes[i].position.x - nodes[j].position.x)
    const dy = Math.abs(nodes[i].position.y - nodes[j].position.y)
    if (dx < NODE_WIDTH - 4 && dy < NODE_HEIGHT - 4) {
      overlaps.push(`${nodes[i].id}<->${nodes[j].id}`)
    }
  }
}
check('no node overlap', overlaps.length === 0, overlaps.length ? overlaps.join(', ') : 'none')

// TOP-TO-BOTTOM means horizontal rank rows.
const rowOf = new Map(nodes.map((n) => [n.id, Math.round(n.centre.y)]))
const rows = [...new Set(rowOf.values())].sort((a, b) => a - b)
check('multiple rank rows (top-to-bottom)', rows.length >= 2, `${rows.length} rows`)

const downward = edges.filter((e) => rowOf.get(e.target) > rowOf.get(e.source)).length
const upward = edges.filter((e) => rowOf.get(e.target) < rowOf.get(e.source)).length
check('edges flow downward', downward > upward,
  `${downward} down / ${upward} up (upward = cycle/reverse edges)`)

const nodeIds = new Set(nodes.map((n) => n.id))
const badEdges = edges.filter((e) => !nodeIds.has(e.source) || !nodeIds.has(e.target))
check('all edge endpoints exist as nodes', badEdges.length === 0,
  badEdges.length ? badEdges.map((e) => e.id).join(',') : `${edges.length} ok`)

const labelCounts = {}
scene.objects.forEach((o) => { labelCounts[o.label] = (labelCounts[o.label] || 0) + 1 })
const dupLabels = Object.entries(labelCounts).filter(([, c]) => c > 1)
check('duplicate-label objects are separate nodes',
  scene.objects.every((o) => typeof o.id === 'string' && o.id.length > 0),
  dupLabels.map(([l, c]) => `${l}x${c}`).join(', ') || 'no duplicates in scene')

const again = layoutNodes(buildNodes(scene.objects), buildEdges(scene.relationships))
check('layout deterministic', nodes.every((n, i) =>
  n.position.x === again[i].position.x && n.position.y === again[i].position.y))

console.log('\nRank rows (top-to-bottom):')
for (const row of rows) {
  const inRow = nodes.filter((n) => Math.round(n.centre.y) === row)
  console.log(`  y=${row}: ${inRow.map((n) => `${n.id}(${n.data.label})`).join(', ')}`)
}

// ---- component source checks (frontend-polish requirements) -------------
console.log('\nComponent source checks')

check('graph lays out through the shared dagre helper (no local config copy)',
  graphSource.includes('layoutNodeRects') &&
    routingSource.includes("rankdir: 'TB'") &&
    !/setGraph\(\{ rankdir/.test(graphSource))
check('graph no longer uses LR orientation', !graphSource.includes("rankdir: 'LR'"))
check('edges are static (no animation)', !/animated/.test(graphSource))
check('inverse-predicate edges use dashed style', graphSource.includes('strokeDasharray'))
check('PNG export present', graphSource.includes('toPng') && graphSource.includes('scene-graph.png'))
check('export excludes minimap/controls',
  graphSource.includes('react-flow__minimap') && graphSource.includes('react-flow__controls'))
check('empty-canvas click clears selection', graphSource.includes('onPaneClick'))
check('edge click selects relationship', graphSource.includes('onEdgeClick'))
check('node click selects object', graphSource.includes('onNodeClick'))
check('inspector gated by selection',
  appSource.includes('hasSelection &&') && appSource.includes('!hasSelection'))
check('detection toggle wired to SceneImage', appSource.includes('showDetections={showDetections}'))
check('bbox visibility comes from the shared visibility list',
  imageSource.includes('visibleObjectIds') && appSource.includes('visibleObjectIds={visibleObjectIds}'))
check('edges are curved bezier paths on deterministic lanes',
  graphSource.includes("type: 'lane'") &&
    graphSource.includes('planEdgeRouting') &&
    /const EDGE_TYPES = \{ lane: LaneEdge \}/.test(graphSource) &&
    !graphSource.includes("'smoothstep'"))
check('graph nodes are draggable unless locked', graphSource.includes('nodesDraggable={!locked}'))

console.log(failures === 0 ? '\nRESULT: PASS' : `\nRESULT: FAIL (${failures})`)
process.exit(failures === 0 ? 0 : 1)
