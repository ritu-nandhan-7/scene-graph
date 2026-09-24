/**
 * LaneEdge - the scene graph's edge: a cubic bezier routed on its own lane with
 * its relationship label placed ON that lane's path.
 *
 * Why a custom edge instead of React Flow's built-in bezier:
 * React Flow computes bezier control points with
 * `distance >= 0 ? 0.5 * distance : curvature * 25 * sqrt(-distance)`.  In a
 * top-to-bottom graph every edge has `distance >= 0`, so `curvature` has no
 * effect and parallel edges between the same two nodes get identical paths and
 * identical label positions.  This component applies the deterministic lane
 * offset / label position computed in src/lib/edgeRouting.js instead, using
 * React Flow's own control-point maths for the base curve (so a lane-0 edge is
 * pixel-identical to the built-in one).
 *
 * Everything the existing UX depends on is preserved:
 *   - the path is still a plain SVG path (BaseEdge: stroke, width, interaction
 *     area for clicking, dash styling for inverse pairs),
 *   - the label is still clickable and selects the relationship,
 *   - label placement only depends on the EDGE, never on the nodes' current
 *     positions, so dragging a node keeps every offset stable while the curves
 *     follow the nodes live.
 *
 * Props come from React Flow (`sourceX` ... `interactionWidth`) plus our own
 * `data` payload: { predicate, confidence, laneOffset, labelT, curvature,
 * labelSelected, labelPoint, onSelect }.  `labelPoint` is the placement decided
 * by SceneGraph for the whole graph (see placeLabels in src/lib/edgeRouting.js),
 * which is how labels also avoid each other - not just the nodes.
 */
import { memo } from 'react'
import { BaseEdge, EdgeLabelRenderer } from '@xyflow/react'
import {
  BASE_CURVATURE,
  DEFAULT_LABEL_T,
  bezierPath,
  edgeLabelPoint,
} from '../lib/edgeRouting.js'

function LaneEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  style,
  data,
  interactionWidth,
}) {
  const geometry = {
    sourceX,
    sourceY,
    targetX,
    targetY,
    curvature: data?.curvature ?? BASE_CURVATURE,
    laneOffset: data?.laneOffset ?? 0,
    labelT: data?.labelT ?? DEFAULT_LABEL_T,
  }

  const path = bezierPath(geometry)
  // SceneGraph places every label in one deterministic pass (nodes + labels as
  // obstacles).  The fallback keeps a lone edge correct if no placement arrived.
  const label = data?.labelPoint || edgeLabelPoint(geometry)
  const selected = Boolean(data?.labelSelected)
  const onSelect = data?.onSelect

  return (
    <>
      <BaseEdge
        id={id}
        path={path}
        style={style}
        interactionWidth={interactionWidth}
      />
      {data?.predicate && (
        <EdgeLabelRenderer>
          <div
            className={`edge-label nodrag nopan${selected ? ' is-selected' : ''}`}
            style={{
              transform: `translate(-50%, -50%) translate(${label.x}px, ${label.y}px)`,
            }}
            data-edge-id={id}
            title={data?.title || data.predicate}
            onClick={onSelect ? () => onSelect(id) : undefined}
          >
            {data.predicate}
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  )
}

export default memo(LaneEdge)
