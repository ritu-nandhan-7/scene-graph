/**
 * Graph visibility + deterministic edge routing verifier.
 *
 * Unlike the older frontend verifiers (which replicate component logic and
 * assert on component source), this one IMPORTS the real modules under test:
 *
 *   src/lib/graphVisibility.js   the one rule that decides what the graph draws
 *   src/lib/edgeRouting.js       the deterministic lane / label plan
 *
 * and runs them against the real scene (tests/_scene.json) plus synthetic
 * stress scenes.  Component source is only inspected where the check is
 * inherently about JSX wiring (which props App passes, that hiding does not
 * re-run dagre, that the lock/search/detection features are untouched).
 *
 * Covers:
 *   A all elements visible by default
 *   B hiding an object hides exactly that node
 *   C hiding an object hides every relationship touching it (both directions)
 *   D unrelated elements stay visible
 *   E hiding one relationship hides exactly that edge
 *   F the other relationships stay visible
 *   G re-showing an object restores its edges except individually hidden ones
 *   H re-showing a relationship restores that edge
 *   J the scene is never mutated (deep-frozen) and search still sees it
 *   K image detections are independent of graph visibility
 *   L lock/drag still driven by `locked`
 *   M hiding never re-lays-out the graph (filter, not relayout)
 *   N counts (visibility summary) are derived, not stored
 *   O deterministic routing: stable, bounded, separated lanes/labels
 *   P label placement on the real layout, through the SAME shared functions
 *     the component renders from (layoutGraphLabels / findLabelClashes /
 *     findLabelsBuriedInNodes) - section P owns no geometry of its own
 *
 * Run from frontend/:  node verify_visibility.mjs
 */
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'
import dagre from '@dagrejs/dagre'
import {
  graphVisibilityCounts,
  isRelationshipGraphVisible,
  resolveSelectionAfterVisibilityChange,
  toggleHiddenId,
  visibleEdgeIds,
  visibleNodeIds,
} from './src/lib/graphVisibility.js'
import {
  BASE_CURVATURE,
  MAX_LABEL_T,
  MAX_LANE_OFFSET,
  MIN_LABEL_T,
  bezierPath,
  boxOverlapArea,
  centeredIndex,
  cubicPoint,
  edgeControlPoints,
  edgeLabelPoint,
  estimateLabelSize,
  fanSpread,
  findInversePartnerIds,
  findLabelClashes,
  findLabelsBuriedInNodes,
  layoutGraphLabels,
  placeLabels,
  planEdgeRouting,
  resolveLabelPlacement,
} from './src/lib/edgeRouting.js'

const here = dirname(fileURLToPath(import.meta.url))
const read = (...parts) => readFileSync(join(here, ...parts), 'utf-8')

const scene = JSON.parse(read('..', 'tests', '_scene.json'))
const appSource = read('src', 'App.jsx')
const graphSource = read('src', 'components', 'SceneGraph.jsx')
const edgeSource = read('src', 'components', 'LaneEdge.jsx')
const inspectorSource = read('src', 'components', 'Inspector.jsx')
const cssSource = read('src', 'App.css')

let failures = 0
function check(name, ok, detail = '') {
  console.log(`  [${ok ? 'PASS' : 'FAIL'}] ${name}${detail ? ' - ' + detail : ''}`)
  if (!ok) failures++
}
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b)

const EMPTY = new Set()
const objects = scene.objects
const relationships = scene.relationships

console.log('GRAPH VISIBILITY + EDGE ROUTING VERIFICATION (real scene)')
console.log(`scene: ${objects.length} objects, ${relationships.length} relationships`)

// ----------------------------------------------------------------------
// A. everything visible by default
// ----------------------------------------------------------------------
console.log('\nA. default state')
check(
  'no hidden ids -> every object node is visible',
  visibleNodeIds(objects, EMPTY).size === objects.length,
  `${visibleNodeIds(objects, EMPTY).size}/${objects.length}`,
)
check(
  'no hidden ids -> every relationship is visible',
  visibleEdgeIds(relationships, EMPTY, EMPTY).size === relationships.length,
  `${visibleEdgeIds(relationships, EMPTY, EMPTY).size}/${relationships.length}`,
)
check(
  'the same edges are visible if the empty sets are plain Sets',
  eq([...visibleEdgeIds(relationships, EMPTY, EMPTY)], relationships.map((r) => r.id)),
)

// ----------------------------------------------------------------------
// B/C/D. hide an object
// ----------------------------------------------------------------------
// Pick the object with the most incident relationships (the worst case).
const incident = (id) =>
  relationships.filter((r) => r.subject_id === id || r.object_id === id).map((r) => r.id)
const busiest = [...objects].sort((a, b) => incident(b.id).length - incident(a.id).length)[0]
const busiestIncident = incident(busiest.id)
const hiddenObjects = new Set([busiest.id])

console.log(`\nB/C/D. hiding object ${busiest.id} (${busiest.label}), ${busiestIncident.length} incident edges`)
const nodesAfter = visibleNodeIds(objects, hiddenObjects)
const edgesAfter = visibleEdgeIds(relationships, hiddenObjects, EMPTY)

check('B. the hidden node is gone', !nodesAfter.has(busiest.id))
check(
  'B. exactly one node disappeared',
  nodesAfter.size === objects.length - 1,
  `${nodesAfter.size} left`,
)
check(
  'C. every incident relationship disappeared, in both directions',
  relationships.every((rel) =>
    rel.subject_id === busiest.id || rel.object_id === busiest.id
      ? !edgesAfter.has(rel.id)
      : true,
  ),
  `removed ${relationships.length - edgesAfter.size}`,
)
check(
  'C. no visible edge still touches the hidden object',
  relationships
    .filter((rel) => edgesAfter.has(rel.id))
    .every((rel) => rel.subject_id !== busiest.id && rel.object_id !== busiest.id),
)
check('D. unrelated objects stay visible', nodesAfter.has(objects[0].id) && nodesAfter.size > 1)
check(
  'D. relationships between two visible objects stay visible',
  relationships
    .filter(
      (rel) =>
        rel.subject_id !== busiest.id &&
        rel.object_id !== busiest.id &&
        edgesAfter.has(rel.id) === true,
    )
    .length === edgesAfter.size,
)

// ----------------------------------------------------------------------
// E/F/H. hide a relationship
// ----------------------------------------------------------------------
// A relationship whose endpoints are both still visible.
const target = relationships.find(
  (rel) => rel.subject_id !== busiest.id && rel.object_id !== busiest.id && edgesAfter.has(rel.id),
)
const hiddenRelationships = new Set([target.id])
const edgesWithRelHidden = visibleEdgeIds(relationships, hiddenObjects, hiddenRelationships)

console.log(`\nE/F/H. hiding relationship ${target.id} (${target.predicate})`)
check('E. the hidden relationship is gone', !edgesWithRelHidden.has(target.id))
check(
  'E. exactly one relationship disappeared',
  edgesWithRelHidden.size === edgesAfter.size - 1,
  `${edgesWithRelHidden.size} left`,
)
check(
  'F. its subject and object are still drawn',
  nodesAfter.has(target.subject_id) && nodesAfter.has(target.object_id),
)
check(
  'F. every other visible relationship is untouched',
  [...edgesAfter].filter((id) => id !== target.id).every((id) => edgesWithRelHidden.has(id)),
)

// ----------------------------------------------------------------------
// G/H. restoring
// ----------------------------------------------------------------------
console.log('\nG. re-showing the object')
const edgesRestored = visibleEdgeIds(relationships, EMPTY, hiddenRelationships)
check(
  'G. exactly the incident relationships came back',
  busiestIncident.every((id) =>
    id === target.id ? !edgesRestored.has(id) : edgesRestored.has(id),
  ),
)
check('G. the individually hidden relationship stays hidden', !edgesRestored.has(target.id))
check(
  'G. nothing else changed',
  edgesRestored.size === relationships.length - 1,
  `${edgesRestored.size}/${relationships.length}`,
)

console.log('\nH. re-showing the relationship')
const edgesAll = visibleEdgeIds(relationships, EMPTY, new Set())
check('H. everything is visible again', edgesAll.size === relationships.length)
check('H. toggling is reversible', eq([...edgesAll], relationships.map((r) => r.id)))

// ----------------------------------------------------------------------
// I. the two hidden sets are independent
// ----------------------------------------------------------------------
console.log('\nI. independence of the two hidden sets')
check(
  'I. hiding a relationship leaves every node drawn',
  visibleNodeIds(objects, new Set([target.id])).size === objects.length,
)
check(
  'I. membership is by id (never by index)',
  isRelationshipGraphVisible(target, EMPTY, new Set([target.id])) === false,
)
check(
  'I. hiding an object and a relationship at once hides both kinds',
  visibleEdgeIds(relationships, new Set([busiest.id]), new Set([target.id])).size <
    relationships.length - 1,
)



// ----------------------------------------------------------------------
// J. the scene is never mutated + search sees everything
// ----------------------------------------------------------------------
console.log('\nJ. scene immutability + search independence')
const snapshot = JSON.stringify(scene)
const frozen = JSON.parse(snapshot) // deep clone frozen instead of the real scene
const deepFreeze = (value) => {
  if (value && typeof value === 'object') {
    Object.values(value).forEach(deepFreeze)
    Object.freeze(value)
  }
  return value
}
deepFreeze(frozen)
let threw = false
try {
  visibleNodeIds(frozen.objects, new Set([busiest.id]))
  visibleEdgeIds(frozen.relationships, new Set([busiest.id]), new Set([target.id]))
  planEdgeRouting(frozen.relationships)
  graphVisibilityCounts(frozen, new Set([busiest.id]), new Set([target.id]))
  toggleHiddenId(new Set(), busiest.id)
} catch (err) {
  threw = true
  console.log(`     threw: ${err.message}`)
}
check('J. no visibility operation mutates the scene (deep-frozen input survived)', !threw)
check('J. the scene JSON is byte-identical afterwards', JSON.stringify(scene) === snapshot)
check(
  'J. search still reads the FULL scene.objects list',
  /scene\.objects\.filter\(\(o\) => o\.label\.toLowerCase\(\)\.includes\(q\)\)/.test(appSource),
)
const searchMemo = (appSource.match(
  /const searchObjectIds = useMemo\([\s\S]*?\n  \}, \[[^\]]*\]\)/,
) || [''])[0]
check(
  'J. search never consults the graph hidden sets',
  searchMemo.length > 0 && !searchMemo.includes('hidden'),
)
check(
  'J. display labels are still built for every object',
  /for \(const obj of scene\.objects\) \{/.test(appSource),
)

// ----------------------------------------------------------------------
// K/L. image detections + lock are untouched by graph visibility
// ----------------------------------------------------------------------
console.log('\nK. image detections + lock unchanged')
check(
  'K. the image panel still receives ALL scene objects',
  /<SceneImage[\s\S]*?objects=\{scene\.objects\}/.test(appSource),
)
const imageVisibilityMemo = (appSource.match(
  /const visibleObjectIds = useMemo\([\s\S]*?\n  \}, \[[^\]]*\]\)/,
) || [''])[0]
check(
  'K. the image visibility list never consults the graph hidden sets',
  imageVisibilityMemo.length > 0 && !imageVisibilityMemo.includes('hidden'),
)
check(
  'K. the detections toggle still owns the image default',
  /if \(showDetections\) return scene\.objects\.map\(\(obj\) => obj\.id\)/.test(appSource),
)
check('L. the lock still drives draggability', /nodesDraggable=\{!locked\}/.test(graphSource))
check('L. the lock button still reports its state', /aria-pressed=\{locked\}/.test(graphSource))

// ----------------------------------------------------------------------
// M. hiding filters the render; it never re-lays-out
// ----------------------------------------------------------------------
console.log('\nM. filtering instead of relayout')
check(
  'M. the drawn nodes are a FILTER of the dragged node state',
  /nodes\.filter\(\(node\) => visibleNodeIdSet\.has\(node\.id\)\)/.test(graphSource),
)
check(
  'M. the drawn edges are a FILTER of the edge state',
  /edges\.filter\(\(edge\) => visibleEdgeIdSet\.has\(edge\.id\)\)/.test(graphSource),
)
check(
  'M. dagre still only runs for a new scene (sceneData memo)',
  /const sceneData = useMemo\([\s\S]*?\[objects, relationships, displayLabels, routing\]/.test(
    graphSource,
  ),
)
check(
  'M. the lane plan comes from the FULL scene, not the visible subset',
  /const routing = useMemo\(\(\) => planEdgeRouting\(relationships\), \[relationships\]\)/.test(
    graphSource,
  ),
)
check(
  'M. React Flow receives the filtered graph + the custom edge type',
  /nodes=\{drawnNodes\}[\s\S]*?edges=\{drawnEdges\}[\s\S]*?edgeTypes=\{EDGE_TYPES\}/.test(
    graphSource,
  ),
)
check(
  'M. dragging still updates positions (onNodesChange wired)',
  /onNodesChange=\{onNodesChange\}/.test(graphSource),
)
check(
  'M. the export still captures the rendered canvas (hidden elements absent)',
  /toPng\(wrapperRef\.current/.test(graphSource),
)


// ----------------------------------------------------------------------
// N. selection cleanup + derived counts (no stale inspector)
// ----------------------------------------------------------------------
console.log('\nN. selection cleanup + summary counts')
const relOfBusiest = relationships.find(
  (r) => r.subject_id === busiest.id || r.object_id === busiest.id,
)
const cleaning = resolveSelectionAfterVisibilityChange({
  scene,
  selectedObjectId: busiest.id,
  selectedRelationshipId: relOfBusiest.id,
  hiddenObjectIds: new Set([busiest.id]),
  hiddenRelationshipIds: new Set(),
})
check(
  'N. hiding the selected object clears the object selection',
  cleaning.selectedObjectId === null,
)
check(
  'N. and clears a relationship selection that touches it',
  cleaning.selectedRelationshipId === null,
)
const keepOther = resolveSelectionAfterVisibilityChange({
  scene,
  selectedObjectId: objects[0].id,
  selectedRelationshipId: null,
  hiddenObjectIds: new Set([busiest.id]),
  hiddenRelationshipIds: new Set(),
})
check(
  'N. hiding an UNRELATED object keeps the selection',
  keepOther.selectedObjectId === objects[0].id,
)
const hideRelSel = resolveSelectionAfterVisibilityChange({
  scene,
  selectedObjectId: null,
  selectedRelationshipId: target.id,
  hiddenObjectIds: new Set(),
  hiddenRelationshipIds: new Set([target.id]),
})
check('N. hiding the selected relationship clears it', hideRelSel.selectedRelationshipId === null)

const counts = graphVisibilityCounts(scene, new Set([busiest.id]), new Set([target.id]))
check(
  'N. counts are derived from the same rule',
  counts.totalObjects === objects.length &&
    counts.totalRelationships === relationships.length &&
    counts.visibleObjects === objects.length - 1 &&
    counts.visibleRelationships ===
      visibleEdgeIds(relationships, new Set([busiest.id]), new Set([target.id])).size,
  `${counts.visibleObjects}/${counts.totalObjects} objects, ${counts.visibleRelationships}/${counts.totalRelationships} relationships`,
)
check(
  'N. the inspector exposes one visibility switch per selection type',
  /id="object-visibility-toggle"/.test(inspectorSource) &&
    /id="relationship-visibility-toggle"/.test(inspectorSource),
)
check(
  'N. the switch is a real checkbox (keyboard accessible)',
  /className="toggle-input"[\s\S]{0,120}?type="checkbox"[\s\S]{0,120}?checked=\{visible\}/.test(
    inspectorSource,
  ),
)
check(
  'N. the graph offers a way back (show-all button)',
  /show-all-visibility/.test(graphSource) && /onShowAllVisibility/.test(graphSource),
)
check(
  'N. the switches reuse the shared, already-tested toggle styling',
  /\.toggle-input:checked \+ \.toggle-track/.test(cssSource),
)


// ----------------------------------------------------------------------
// O. deterministic routing: stable, bounded, separated
// ----------------------------------------------------------------------
console.log('\nO. deterministic edge routing / label placement')
const plan = planEdgeRouting(relationships)
check('O. every relationship is planned', Object.keys(plan).length === relationships.length)
check('O. the plan is deterministic (two calls are identical)', eq(plan, planEdgeRouting(relationships)))
check(
  'O. every lane offset + label position is bounded and finite',
  Object.values(plan).every(
    (p) =>
      Number.isFinite(p.laneOffset) &&
      Math.abs(p.laneOffset) <= MAX_LANE_OFFSET &&
      p.labelT >= MIN_LABEL_T &&
      p.labelT <= MAX_LABEL_T &&
      p.curvature > 0,
  ),
)
check(
  'O. a lone relationship keeps the default midpoint (existing UX unchanged)',
  Object.values(plan).some((p) => p.pairCount === 1 && p.labelT === 0.5 && p.laneOffset === 0),
)

const parallelGroups = new Map()
relationships.forEach((rel) => {
  const key = `${rel.subject_id}->${rel.object_id}`
  if (!parallelGroups.has(key)) parallelGroups.set(key, [])
  parallelGroups.get(key).push(rel.id)
})
const parallel = [...parallelGroups.values()].filter((ids) => ids.length > 1)
console.log(
  `     ${parallel.length} parallel group(s) holding ${parallel.reduce((n, ids) => n + ids.length, 0)} edges`,
)
check(
  'O. parallel edges get pairwise-distinct lane offsets',
  parallel.every((ids) => new Set(ids.map((id) => plan[id].laneOffset)).size === ids.length),
)
check(
  'O. parallel edges get pairwise-distinct label positions along the curve',
  parallel.every((ids) => new Set(ids.map((id) => plan[id].labelT)).size === ids.length),
)
check(
  'O. lane order follows scene order (lanes can never swap between renders)',
  parallel.every((ids) =>
    ids.every((id, index) => index === 0 || plan[ids[index - 1]].laneOffset < plan[id].laneOffset),
  ),
)

// Synthetic stress scene: 4 parallel edges, a fan-out, an inverse pair - all
// with identical geometry, so any shared lane would be a visible overlap.
const stress = []
for (let i = 0; i < 4; i++) {
  stress.push({
    id: `s_p${i}`,
    subject_id: 'obj_0',
    object_id: 'obj_1',
    predicate: `p${i}`,
    confidence: 0.5 + i / 100,
  })
}
stress.push({ id: 's_fan1', subject_id: 'obj_0', object_id: 'obj_2', predicate: 'a', confidence: 0.9 })
stress.push({ id: 's_fan2', subject_id: 'obj_0', object_id: 'obj_3', predicate: 'b', confidence: 0.9 })
stress.push({ id: 's_inv1', subject_id: 'obj_5', object_id: 'obj_6', predicate: 'wearing', confidence: 0.9 })
stress.push({ id: 's_inv2', subject_id: 'obj_6', object_id: 'obj_5', predicate: 'worn by', confidence: 0.9 })

const stressPlan = planEdgeRouting(stress)
// All six obj_0 edges share the exact same corridor, and obj_1 sits directly
// below obj_0, so every "dodge" has to come from the lanes themselves.
const stressPlacement = placeLabels({
  edges: stress.map((rel) => ({
    id: rel.id,
    source: rel.subject_id,
    target: rel.object_id,
    predicate: rel.predicate,
  })),
  plan: stressPlan,
  nodeRects: {
    obj_0: { x: 0, y: 0, w: 170, h: 56 },
    obj_1: { x: 0, y: 400, w: 170, h: 56 },
    obj_2: { x: 420, y: 400, w: 170, h: 56 },
    obj_3: { x: 840, y: 400, w: 170, h: 56 },
    obj_5: { x: 0, y: 800, w: 170, h: 56 },
    obj_6: { x: -420, y: 0, w: 170, h: 56 },
  },
})
const stressBoxes = stress.map((rel) => ({ id: rel.id, ...stressPlacement[rel.id] }))
const stressClashes = []
for (let i = 0; i < stressBoxes.length; i++) {
  for (let j = i + 1; j < stressBoxes.length; j++) {
    const a = stressBoxes[i]
    const b = stressBoxes[j]
    const area = boxOverlapArea(a, b)
    if (area > 0.05 * Math.min(a.w * a.h, b.w * b.h)) {
      stressClashes.push(`${a.id}/${b.id}`)
    }
  }
}

check(
  'O. stress: all 4 parallel lanes are distinct',
  new Set([0, 1, 2, 3].map((i) => stressPlan[`s_p${i}`].laneOffset)).size === 4,
)
check(
  'O. stress: the inverse pair takes opposite lanes',
  stressPlan.s_inv1.laneOffset !== 0 &&
    Math.sign(stressPlan.s_inv1.laneOffset) === -Math.sign(stressPlan.s_inv2.laneOffset),
  `${stressPlan.s_inv1.laneOffset} vs ${stressPlan.s_inv2.laneOffset}`,
)
check(
  'O. stress: the inverse pair is detected as a pair',
  findInversePartnerIds(stress).s_inv1 === 's_inv2' &&
    stressPlan.s_inv1.hasInverse === true &&
    stressPlan.s_inv2.inversePartnerId === 's_inv1',
)
check(
  'O. stress: 8 labels in one corridor, none overlapping',
  stressClashes.length === 0,
  stressClashes.length ? stressClashes.join(', ') : '0 clashes',
)
check(
  'O. stress: every label stays on its own edge (t within bounds)',
  stressBoxes.every((b) => b.t >= MIN_LABEL_T && b.t <= MAX_LABEL_T),
)
check(
  'O. stress: lane offsets stay bounded',
  Object.values(stressPlan).every((p) => Math.abs(p.laneOffset) <= MAX_LANE_OFFSET),
)

// Geometry: identical maths to React Flow, plus the lane shift.
const geometry = { sourceX: 0, sourceY: 0, targetX: 0, targetY: 200 }
const cps = edgeControlPoints({ ...geometry, curvature: BASE_CURVATURE, laneOffset: 0 })
check(
  'O. lane 0 reproduces React Flow control points exactly (0.5 * distance)',
  cps.c1.x === 0 && cps.c1.y === 100 && cps.c2.x === 0 && cps.c2.y === 100,
  JSON.stringify(cps),
)
check(
  'O. a lane offset moves the curve sideways, never the endpoints',
  (() => {
    const m = bezierPath({ ...geometry, laneOffset: 30 }).match(/^M(-?[\d.]+),(-?[\d.]+) C/)
    return Number(m[1]) === 0 && Number(m[2]) === 0
  })(),
)
check(
  'O. the label sits on its own curve (t = 0.5 midpoint sanity check)',
  (() => {
    const p = edgeLabelPoint({ ...geometry, laneOffset: 0, labelT: 0.5 })
    const c = cubicPoint(0.5, { x: 0, y: 0 }, { x: 0, y: 100 }, { x: 0, y: 100 }, { x: 0, y: 200 })
    return Math.abs(p.x - c.x) < 1e-9 && Math.abs(p.y - c.y) < 1e-9
  })(),
)
check(
  'O. centredIndex is the documented stable ranking',
  centeredIndex(0, 1) === 0 &&
    centeredIndex(0, 2) === -0.5 &&
    centeredIndex(1, 2) === 0.5 &&
    centeredIndex(0, 3) === -1 &&
    centeredIndex(2, 3) === 1,
)
check(
  'O. fanSpread is injective per group and bounded (no clamped collisions)',
  (() => {
    const spread = [0, 1, 2, 3, 4, 5].map((rank) => fanSpread(rank, 6))
    return (
      new Set(spread).size === 6 &&
      spread.every((value) => Math.abs(value) <= 2) &&
      fanSpread(0, 1) === 0
    )
  })(),
)
check(
  'O. the custom edge uses the shared lib (no second copy of the maths)',
  /bezierPath|edgeLabelPoint/.test(edgeSource) &&
    /from '\.\.\/lib\/edgeRouting\.js'/.test(edgeSource) &&
    !/pathOptions/.test(graphSource),
)
check(
  'O. labels render in the edge label layer with their own plate',
  /EdgeLabelRenderer/.test(edgeSource) &&
    /edge-label/.test(edgeSource) &&
    /\.graph-container \.edge-label \{/.test(cssSource),
)
check(
  'O. clicking a label still selects the relationship',
  /onClick=\{onSelect \? \(\) => onSelect\(id\) : undefined\}/.test(edgeSource),
)
check(
  'O. each lane edge keeps a wide click target',
  /interactionWidth=\{interactionWidth\}/.test(edgeSource),
)

// ----------------------------------------------------------------------
// P. real layout: no label may overlap another label or sit under a node
// ----------------------------------------------------------------------
// The geometry below reuses the shared lib (same dagre config, same node
// size, same bottom -> top ports as SceneGraph.jsx); the wiring check at the
// end of this section fails loudly if the component ever stops importing them.
console.log('\nP. label placement on the REAL dagre layout')
const predicates = new Map(relationships.map((r) => [r.id, r.predicate]))

function realLayout(hiddenNodes = new Set(), hiddenRels = new Set()) {
  const visibleObjects = objects.filter((o) => !hiddenNodes.has(o.id))
  const edges = relationships.filter((r) => isRelationshipGraphVisible(r, hiddenNodes, hiddenRels))
  // Same shared pipeline the component renders from: node rects + one
  // deterministic placement pass over the visible edges.
  const { nodeRects, placement } = layoutGraphLabels({
    objects: visibleObjects,
    relationships: edges,
    dagreModule: dagre,
  })
  const boxes = edges.map((rel) => ({
    id: rel.id,
    predicate: rel.predicate,
    ...placement[rel.id],
  }))
  return { rects: nodeRects, boxes, placement }
}

function measureLayout(hiddenNodes, hiddenRels) {
  const { rects, boxes, placement } = realLayout(hiddenNodes, hiddenRels)
  // Same shared reporters the lib ships: no local copy of the thresholds.
  const clashes = findLabelClashes(placement).map(
    ({ a, b, ratio: r }) =>
      `${predicates.get(a)} (${a}) vs ${predicates.get(b)} (${b}) ${(r * 100).toFixed(0)}%`,
  )
  const buried = findLabelsBuriedInNodes(placement, rects)
  return { boxes, clashes, buried }
}

const fullLayout = measureLayout(new Set(), new Set())
console.log(
  `     ${fullLayout.boxes.length} labels; ${fullLayout.clashes.length} clash(es); ${fullLayout.buried.length} hidden behind a node`,
)
fullLayout.clashes.slice(0, 10).forEach((c) => console.log(`       clash: ${c}`))
if (fullLayout.buried.length) console.log(`       buried: ${fullLayout.buried.join(', ')}`)

check('P. no two relationship labels overlap in the real layout', fullLayout.clashes.length === 0)
check('P. no relationship label is hidden behind a node', fullLayout.buried.length === 0)

const sparseLayout = measureLayout(new Set([busiest.id]), new Set([target.id]))
sparseLayout.clashes.forEach((c) => console.log(`       hidden-state clash: ${c}`))
if (sparseLayout.buried.length) {
  console.log(`       hidden-state buried: ${sparseLayout.buried.join(', ')}`)
}
check(
  'P. the same holds while elements are hidden',
  sparseLayout.clashes.length === 0 && sparseLayout.buried.length === 0,
  `${sparseLayout.boxes.length} labels`,
)

// The shared geometry must keep matching the component's wiring: the lib owns
// the numbers, the component imports them (no local copies).
const edgeRoutingSource = read('src', 'lib', 'edgeRouting.js')
check(
  'P. the layout mirror matches SceneGraph (shared dagre config + node size)',
  /rankdir: 'TB', nodesep: 70, ranksep: 130, marginx: 30, marginy: 30/.test(edgeRoutingSource) &&
    /export const NODE_WIDTH = 170\b/.test(edgeRoutingSource) &&
    /export const NODE_HEIGHT = 56\b/.test(edgeRoutingSource) &&
    /layoutNodeRects/.test(graphSource) &&
    /from '\.\.\/lib\/edgeRouting\.js'/.test(graphSource) &&
    !/const NODE_WIDTH = 170/.test(graphSource),
)
check(
  'P. the label box model only assumes what the CSS guarantees',
  /\.graph-container \.edge-label \{[\s\S]*?font-size: 10\.5px[\s\S]*?white-space: nowrap/.test(
    cssSource,
  ),
)

// ----------------------------------------------------------------------
console.log(
  `\nGraph visibility + routing: ${failures === 0 ? 'ALL CHECKS PASSED' : `${failures} FAILED`}`,
)
process.exit(failures === 0 ? 0 : 1)

