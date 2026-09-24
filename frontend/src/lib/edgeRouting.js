/**
 * Deterministic edge routing + label placement for the scene graph.
 *
 * WHY THIS EXISTS
 * ---------------
 * React Flow's built-in bezier edge puts every label at the curve midpoint
 * (getBezierEdgeCenter) and computes its control points with
 * `calculateControlOffset(distance, curvature) = distance >= 0 ? 0.5 * distance
 * : curvature * 25 * sqrt(-distance)`.  For a top-to-bottom graph every edge
 * has `distance = targetY - sourceY >= 0`, so the `curvature` option is INERT
 * and two relationships sharing both endpoints get byte-identical paths - their
 * labels land on exactly the same pixel.  That was the visible
 * "wearing / sitting on / holding" pile-up.
 *
 * This module therefore computes, for every relationship, ONE deterministic
 * lane offset (px, perpendicular to the edge) and ONE label position along its
 * own curve, derived only from the scene data: relationship order, endpoint
 * order and group sizes.  Nothing is random and nothing depends on render
 * order, so:
 *
 *   - the same scene always routes the same way,
 *   - two edges can never swap lanes between renders,
 *   - hiding/showing graph elements does NOT re-lane the remaining edges,
 *   - dragging a node keeps every lane offset (only the geometry changes).
 *
 * There is no graph library here and no dependency was added: the paths are
 * plain cubic beziers built with React Flow's own control-point formula.
 *
 * Groups (all in scene order, so every rank is stable):
 *   pair    same ordered subject -> object   true parallel edges
 *   source  same subject                     fan-out
 *   target  same object                      fan-in
 *   inverse A->B and B->A with inverse predicates
 */

/** Gentle curve used by both the built-in and the lane edge. */
export const BASE_CURVATURE = 0.32

/** Perpendicular distance between two parallel edges of the same pair. */
export const LANE_GAP = 34
/** Spread for edges leaving one subject (fan-out) / entering one object (fan-in). */
export const FAN_OUT_GAP = 22
export const FAN_IN_GAP = 22

/** How far a label slides ALONG its own curve, per lane / fan step. */
export const LABEL_T_LANE_STEP = 0.1
export const LABEL_T_FAN_STEP = 0.06
export const DEFAULT_LABEL_T = 0.5

/** Hard bounds: lanes stay gentle, labels stay on the middle of their edge. */
export const MAX_LANE_OFFSET = 110
export const MIN_LABEL_T = 0.28
export const MAX_LABEL_T = 0.72
export const CURVATURE_LANE_STEP = 0.06
export const MIN_CURVATURE = 0.1
export const MAX_CURVATURE = 0.7

/**
 * Label dodging: when the lane's label would sit behind a node, the label is
 * slid along its own curve until it is clear.  Tried in this order (0 first, so
 * a label that is already clear never moves), and the best candidate wins if
 * every one of them is blocked.  Purely a function of the current geometry, so
 * it stays stable while dragging.
 */
export const LABEL_DODGE_OFFSETS = [0, -0.08, 0.08, -0.16, 0.16, -0.24, 0.24]

/**
 * Safety margins inside the placement pass.  The label box is an estimate of a
 * real DOM rectangle, so a few pixels of margin guarantee that a candidate the
 * planner calls "clear" is also visually clear in the browser.
 */
export const NODE_OBSTACLE_PADDING = 6
export const LABEL_OBSTACLE_PADDING = 2

// Inverse predicates from the Visual Genome predicate dictionary (unchanged).
export const INVERSE_PREDICATES = { wearing: 'worn by', 'worn by': 'wearing' }

export function clamp(value, lo, hi) {
  return Math.min(hi, Math.max(lo, value))
}

/** Centred rank: count 1 -> 0, count 2 -> -0.5/+0.5, count 3 -> -1/0/1. */
export function centeredIndex(rank, count) {
  return count <= 1 ? 0 : rank - (count - 1) / 2
}

/**
 * Bounded fan spread: centred rank -> sqrt-shaped value
 * (-2.5 -> -1.58, -1.5 -> -1.22, -0.5 -> -0.71, 0.5 -> 0.71, ...).
 *
 * A plain clamp would make two edges of a large fan share a lane (the bug this
 * replaces), while the raw centred rank grows without bound.  sqrt is injective
 * per group (so no two edges of one hub can collide) and grows slowly enough
 * that even a 20-edge hub stays well inside MAX_LANE_OFFSET.
 */
export function fanSpread(rank, count) {
  const centred = centeredIndex(rank, count)
  return Math.sign(centred) * Math.sqrt(Math.abs(centred))
}

/** Maps relationship id -> id of its inverse partner (swapped endpoints). */
export function findInversePartnerIds(relationships) {
  const partners = {}
  if (!relationships) return partners
  for (const a of relationships) {
    for (const b of relationships) {
      if (a.id === b.id) continue
      if (
        a.subject_id === b.object_id &&
        a.object_id === b.subject_id &&
        INVERSE_PREDICATES[a.predicate] === b.predicate
      ) {
        partners[a.id] = b.id
        break
      }
    }
  }
  return partners
}

// Graph geometry: the node footprint and the dagre layout.  Exported so the
// component AND the verifier lay out the same boxes from the same numbers -
// there is exactly one dagre config, one node size, one port anchor.
export const NODE_WIDTH = 170
export const NODE_HEIGHT = 56
export const DAGRE_CONFIG = { rankdir: 'TB', nodesep: 70, ranksep: 130, marginx: 30, marginy: 30 }

/**
 * Layout node rectangles in flow coordinates via dagre.  `dagreModule` is the
 * dagre instance (passed in so this pure lib never imports a graph library);
 * SceneGraph.jsx passes its bundled dagre, the verifier passes its own.
 *
 * Returns object id -> { x, y, w, h } (top-left corner, like React Flow).
 */
export function layoutNodeRects(objects, relationships, dagreModule) {
  const rects = {}
  if (!objects || objects.length === 0) return rects
  const graph = new dagreModule.graphlib.Graph({ multigraph: true })
  graph.setGraph({ ...DAGRE_CONFIG })
  graph.setDefaultEdgeLabel(() => ({}))
  objects.forEach((obj) => graph.setNode(obj.id, { width: NODE_WIDTH, height: NODE_HEIGHT }))
  ;(relationships || []).forEach((rel, index) =>
    graph.setEdge(rel.subject_id, rel.object_id, { weight: 1 }, rel.id || `edge_${index}`),
  )
  dagreModule.layout(graph)
  // dagre reports node centres; React Flow positions are top-left corners.
  for (const obj of objects) {
    const centre = graph.node(obj.id)
    rects[obj.id] = {
      x: centre.x - NODE_WIDTH / 2,
      y: centre.y - NODE_HEIGHT / 2,
      w: NODE_WIDTH,
      h: NODE_HEIGHT,
    }
  }
  return rects
}

/** Centre-top port position for an edge endpoint (mirrors SceneGraph ports). */
export function edgePortPoint(rect, isSource) {
  return isSource
    ? { x: rect.x + rect.w / 2, y: rect.y + rect.h }
    : { x: rect.x + rect.w / 2, y: rect.y }
}

/**
 * Full label layout for a (possibly visibility-filtered) graph: node rects +
 * one deterministic label-placement pass.  The component and the verifier both
 * call this instead of each re-implementing the pipeline.
 */
export function layoutGraphLabels({ objects, relationships, dagreModule }) {
  const routing = planEdgeRouting(relationships)
  const nodeRects = layoutNodeRects(objects, relationships, dagreModule)
  const placement = placeLabels({
    edges: (relationships || []).map((rel) => ({
      id: rel.id,
      source: rel.subject_id,
      target: rel.object_id,
      predicate: rel.predicate,
    })),
    plan: routing,
    nodeRects,
  })
  return { routing, nodeRects, placement }
}

/**
 * Label-overlap report over a placement: [{ a, b, ratio }] for every pair of
 * labels whose boxes overlap by more than `threshold` of the smaller box.
 * The threshold matches the placement's own dodge rule (0 overlap wins).
 */
export function findLabelClashes(placement, threshold = 0.2) {
  const boxes = Object.entries(placement || {}).map(([id, box]) => ({ id, ...box }))
  const clashes = []
  for (let i = 0; i < boxes.length; i++) {
    for (let j = i + 1; j < boxes.length; j++) {
      const area = boxOverlapArea(boxes[i], boxes[j])
      if (area === 0) continue
      const r = area / Math.min(boxes[i].w * boxes[i].h, boxes[j].w * boxes[j].h)
      if (r > threshold) clashes.push({ a: boxes[i].id, b: boxes[j].id, ratio: r })
    }
  }
  return clashes
}

/**
 * Labels whose centre sits inside a node rect AND whose box overlaps that
 * node by >30% of the smaller area.  Centre-in-node mirrors the browser
 * hit-test (verify_browser.mjs: elementFromPoint + closest('.react-flow__node'))
 * so the headless and live checks agree by construction.
 */
export function findLabelsBuriedInNodes(placement, nodeRects, threshold = 0.3) {
  const buried = []
  for (const [id, box] of Object.entries(placement || {})) {
    const cx = box.x + box.w / 2
    const cy = box.y + box.h / 2
    for (const rect of Object.values(nodeRects || {})) {
      const centreInside =
        cx >= rect.x && cx <= rect.x + rect.w && cy >= rect.y && cy <= rect.y + rect.h
      if (!centreInside) continue
      const area = boxOverlapArea(box, rect)
      const r = area === 0 ? 0 : area / Math.min(box.w * box.h, rect.w * rect.h)
      if (r > threshold) {
        buried.push(id)
        break
      }
    }
  }
  return buried
}

function groupBy(relationships, keyOf) {
  const groups = new Map()
  relationships.forEach((rel) => {
    const key = keyOf(rel)
    if (!groups.has(key)) groups.set(key, [])
    groups.get(key).push(rel.id)
  })
  return groups
}

// ----------------------------------------------------------------------
// Routing plan
// ----------------------------------------------------------------------

/**
 * The routing plan: relationship id -> { laneOffset, labelT, curvature, ... }.
 *
 * Pure function of the relationship list (scene order).  Called once per scene
 * via useMemo in SceneGraph.jsx, and directly by the verifier.
 */
export function planEdgeRouting(relationships) {
  const plan = {}
  if (!relationships || relationships.length === 0) return plan

  const pairGroups = groupBy(
    relationships,
    // UNORDERED pair: A->B and B->A share the same corridor between the two
    // nodes and cross each other there, so they must be laned together.
    (rel) =>
      [rel.subject_id, rel.object_id].sort().join('..'),
  )
  const sourceGroups = groupBy(relationships, (rel) => rel.subject_id)
  const targetGroups = groupBy(relationships, (rel) => rel.object_id)
  const partners = findInversePartnerIds(relationships)

  relationships.forEach((rel) => {
    const pairList = pairGroups.get([rel.subject_id, rel.object_id].sort().join('..'))
    const sourceList = sourceGroups.get(rel.subject_id)
    const targetList = targetGroups.get(rel.object_id)

    const pairRank = pairList.indexOf(rel.id)
    const lane = centeredIndex(pairRank, pairList.length)
    const fanOut = fanSpread(sourceList.indexOf(rel.id), sourceList.length)
    const fanIn = fanSpread(targetList.indexOf(rel.id), targetList.length)

    const inversePartnerId = partners[rel.id] || null

    const laneOffset = clamp(
      lane * LANE_GAP + fanOut * FAN_OUT_GAP - fanIn * FAN_IN_GAP,
      -MAX_LANE_OFFSET,
      MAX_LANE_OFFSET,
    )

    const labelT = clamp(
      DEFAULT_LABEL_T +
        lane * LABEL_T_LANE_STEP +
        fanOut * LABEL_T_FAN_STEP -
        fanIn * LABEL_T_FAN_STEP,
      MIN_LABEL_T,
      MAX_LABEL_T,
    )

    plan[rel.id] = {
      lane,
      laneOffset,
      labelT,
      curvature: clamp(
        BASE_CURVATURE + lane * CURVATURE_LANE_STEP,
        MIN_CURVATURE,
        MAX_CURVATURE,
      ),
      hasInverse: Boolean(inversePartnerId),
      inversePartnerId,
      pairRank,
      pairCount: pairList.length,
      sourceCount: sourceList.length,
      targetCount: targetList.length,
    }
  })

  return plan
}

// ----------------------------------------------------------------------
// Geometry (identical maths to React Flow's own bottom -> top bezier)
// ----------------------------------------------------------------------

/** React Flow's calculateControlOffset: curvature only matters for upward edges. */
export function controlOffset(distance, curvature) {
  if (distance >= 0) return 0.5 * distance
  return curvature * 25 * Math.sqrt(-distance)
}

/**
 * Cubic control points for a source-bottom -> target-top edge, shifted by the
 * lane offset perpendicular to the straight source->target line.  The endpoints
 * stay anchored to the node ports, so parallel edges fan out mid-way while the
 * ports still meet the nodes exactly.
 */
export function edgeControlPoints({
  sourceX,
  sourceY,
  targetX,
  targetY,
  curvature = BASE_CURVATURE,
  laneOffset = 0,
}) {
  const offset = controlOffset(targetY - sourceY, curvature)
  const dx = targetX - sourceX
  const dy = targetY - sourceY
  const length = Math.hypot(dx, dy) || 1
  // Unit normal of the edge direction.  For a vertical edge this is horizontal,
  // which is exactly the axis parallel edges have to separate along.
  const nx = (-dy / length) * laneOffset
  const ny = (dx / length) * laneOffset

  return {
    c1: { x: sourceX + nx, y: sourceY + offset + ny },
    c2: { x: targetX + nx, y: targetY - offset + ny },
  }
}

/** SVG path string for one routed edge. */
export function bezierPath(geometry) {
  const { c1, c2 } = edgeControlPoints(geometry)
  return `M${geometry.sourceX},${geometry.sourceY} C${c1.x},${c1.y} ${c2.x},${c2.y} ${geometry.targetX},${geometry.targetY}`
}

/** Point on a cubic bezier at parameter t. */
export function cubicPoint(t, p0, p1, p2, p3) {
  const mt = 1 - t
  const a = mt * mt * mt
  const b = 3 * mt * mt * t
  const c = 3 * mt * t * t
  const d = t * t * t
  return {
    x: a * p0.x + b * p1.x + c * p2.x + d * p3.x,
    y: a * p0.y + b * p1.y + c * p2.y + d * p3.y,
  }
}

/**
 * Label position: ON the edge's own path, at the lane-dependent parameter t.
 * t = 0.5 reproduces React Flow's usual midpoint, so a lone edge is unchanged.
 */
export function edgeLabelPoint(geometry) {
  const { c1, c2 } = edgeControlPoints(geometry)
  const t = geometry.labelT ?? DEFAULT_LABEL_T
  return cubicPoint(
    t,
    { x: geometry.sourceX, y: geometry.sourceY },
    c1,
    c2,
    { x: geometry.targetX, y: geometry.targetY },
  )
}

// ----------------------------------------------------------------------
// Label boxes + dodging around nodes
// ----------------------------------------------------------------------

/**
 * Conservative label box for a predicate.  Kept in one place so the component
 * and the verifiers measure the same box; verify_browser.mjs additionally
 * measures the REAL rendered rectangles in a live browser.
 */
export function estimateLabelSize(predicate) {
  const text = predicate || ''
  return {
    w: Math.min(190, Math.max(30, text.length * 6.3 + 16)),
    h: 18,
  }
}

/** Label box at a specific t (t defaults to the lane's own labelT). */
export function labelBoxAt(geometry, t, size) {
  const point = edgeLabelPoint({ ...geometry, labelT: t ?? geometry.labelT })
  const w = size?.w ?? 0
  const h = size?.h ?? 0
  return { x: point.x - w / 2, y: point.y - h / 2, w, h, x0: point.x, y0: point.y }
}

/** Overlapping area of two boxes (0 when they are disjoint). */
export function boxOverlapArea(a, b) {
  const w = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x)
  const h = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y)
  return w <= 0 || h <= 0 ? 0 : w * h
}

/**
 * Final label placement for one edge.
 *
 * Starts from the lane's own t and, if the label box would be covered by an
 * obstacle (node rectangles, or labels already placed by placeLabels), slides
 * along the curve using the fixed LABEL_DODGE_OFFSETS ladder.  Deterministic:
 * the same geometry always yields the same placement, and a label that is
 * already clear never moves (offset 0 is tried first).
 */
export function resolveLabelPlacement({ geometry, obstacles, size }) {
  const list = obstacles || []
  let best = null
  for (const offset of LABEL_DODGE_OFFSETS) {
    const t = clamp((geometry.labelT ?? DEFAULT_LABEL_T) + offset, MIN_LABEL_T, MAX_LABEL_T)
    const box = labelBoxAt(geometry, t, size)
    let covered = 0
    for (const rect of list) covered += boxOverlapArea(box, rect)
    if (covered === 0) return { ...box, t }
    if (!best || covered < best.covered) best = { ...box, covered, t }
  }
  return best || { ...labelBoxAt(geometry, geometry.labelT, size), t: geometry.labelT }
}

/**
 * Places EVERY relationship label of a graph in one deterministic pass.
 *
 * Priority is edge order (scene order), and each label is dodged around the
 * nodes plus the labels that were already placed before it.  Because the order
 * never depends on render timing, the same scene always produces the same
 * label positions, and a later edge can never push an earlier one.
 *
 * @param edges     [{ id, source, target, predicate }] - visible edges, in order
 * @param plan      relationship id -> routing plan entry (planEdgeRouting)
 * @param nodeRects object id -> { x, y, w, h } in flow coordinates
 * @returns         edge id -> { x, y, w, h, t } (x/y = label box top-left)
 */
export function placeLabels({ edges, plan, nodeRects }) {
  const placement = {}
  if (!edges || edges.length === 0) return placement
  const rects = nodeRects || {}

  // Obstacles with safety margins: a candidate the planner calls "clear" must
  // also be visually clear of the real DOM rectangles.
  const nodes = Object.values(rects).map((r) => inflate(r, NODE_OBSTACLE_PADDING))
  const placed = [] // label boxes, inflated by LABEL_OBSTACLE_PADDING when stored

  for (const edge of edges) {
    const route = plan?.[edge.id]
    const source = rects[edge.source]
    const target = rects[edge.target]
    if (!route || !source || !target) continue

    // Same port maths as the node component: bottom centre -> top centre.
    const geometry = {
      sourceX: source.x + source.w / 2,
      sourceY: source.y + source.h,
      targetX: target.x + target.w / 2,
      targetY: target.y,
      ...route,
    }
    const size = estimateLabelSize(edge.predicate)
    const box = resolveLabelPlacement({
      geometry,
      obstacles: nodes.concat(placed),
      size,
    })
    placement[edge.id] = { x: box.x, y: box.y, w: size.w, h: size.h, t: box.t, x0: box.x0, y0: box.y0 }
    placed.push(inflate({ x: box.x, y: box.y, w: size.w, h: size.h }, LABEL_OBSTACLE_PADDING))
  }

  return placement
}

/** Expands a box by `pad` px on every side. */
function inflate(box, pad) {
  return { x: box.x - pad, y: box.y - pad, w: box.w + pad * 2, h: box.h + pad * 2 }
}
