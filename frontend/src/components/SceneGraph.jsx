/**
 * SceneGraph - React Flow view of one scene.
 *
 *   one scene object  -> one node   (node id = object id, e.g. "obj_0")
 *   one relationship  -> one edge   (edge id = relationship id, e.g. "rel_0")
 *
 * LAYOUT: dagre, TOP-TO-BOTTOM (rankdir "TB"), so a relationship reads
 * subject -> predicate -> object downwards.  Deterministic: the same scene
 * always lays out the same way.
 *
 * INVERSE PREDICATES: "wearing" (VG 48) and "worn by" (VG 50) are inverse
 * entries of the project's predicate dictionary.  Both relationships are kept
 * as separate, independently selectable edges - nothing is merged or deleted -
 * but the inverse partner is drawn dashed and shares an `inversePartnerId` in
 * edge data so the pair reads as one visual group instead of two
 * identical-looking edges.
 *
 * EDGES are cubic bezier paths routed through the deterministic lane plan in
 * src/lib/edgeRouting.js (see LaneEdge.jsx): a relationship label never sits on
 * top of another relationship's label, parallel edges between the same pair fan
 * out into lanes, and inverse pairs get opposite lanes so both stay readable.
 * All predicted relationships remain present and selectable.
 *
 * VISIBILITY: `hiddenObjectIds` / `hiddenRelationshipIds` (owned by App) decide
 * what is DRAWN.  The rule lives in src/lib/graphVisibility.js and is applied
 * here once for nodes and once for edges - nothing is ever deleted from the
 * scene, and node positions survive hiding/showing.
 *
 * DRAGGING: the graph is UNLOCKED by default, so nodes can be rearranged by
 * hand.  Positions live in this component's node state and are therefore kept
 * across selection, search and restyling changes - only a NEW scene (App
 * remounts the explorer via `key={sceneKey}`) starts from a fresh dagre
 * layout.  While `locked` is true React Flow stops node dragging; pan and zoom
 * keep working.  The lock is pure UI state and never touches the scene data.
 *
 * LABELS come from `displayLabels` (object id -> "person 2" when a scene has
 * duplicate labels).  Node ids always remain the raw object ids.
 *
 * The component never calls the API; it only receives scene data.
 */
import { useCallback, useEffect, useMemo, useRef } from 'react'
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  ReactFlowProvider,
  useReactFlow,
} from '@xyflow/react'
import dagre from '@dagrejs/dagre'
import { toPng } from 'html-to-image'
import LaneEdge from './LaneEdge.jsx'
import {
  NODE_HEIGHT,
  NODE_WIDTH,
  layoutNodeRects,
  planEdgeRouting,
  placeLabels,
} from '../lib/edgeRouting.js'
import { visibleEdgeIds, visibleNodeIds } from '../lib/graphVisibility.js'
import '@xyflow/react/dist/style.css'

const COLOR_NODE_BORDER = '#4a90d9'
const COLOR_SELECTED = '#ef4444'
const COLOR_REL_ENDPOINT = '#f0932c'
const COLOR_SEARCH = '#00897b'

// Our custom edge: bezier on a deterministic lane, label on its own path.
const EDGE_TYPES = { lane: LaneEdge }

// Shared empty set: the default when a caller passes no hidden ids.
const EMPTY_SET = new Set()

// Edge label text for the tooltip (the visible text stays the bare predicate).
function edgeTitle(predicate, confidence) {
  if (confidence == null) return predicate
  return `${predicate} - ${(confidence * 100).toFixed(2)}%`
}

const EDGE_COLOR = '#64748b'
const EDGE_COLOR_SELECTED = '#ef4444'
const EDGE_DEFAULT_WIDTH = 1.5

// `displayLabels` maps object id -> human readable label ("person 2" when the
// scene contains more than one "person").  The node id stays the raw object id.
function buildNodes(objects, displayLabels) {
  if (!objects || objects.length === 0) return []
  const labels = displayLabels || {}
  return objects.map((obj) => ({
    id: obj.id,
    data: { label: labels[obj.id] || obj.label, confidence: obj.confidence },
    sourcePosition: 'bottom',
    targetPosition: 'top',
    style: {
      width: NODE_WIDTH,
      height: NODE_HEIGHT,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      textAlign: 'center',
      borderRadius: 6,
      border: `2px solid ${COLOR_NODE_BORDER}`,
      backgroundColor: '#1e2a3a',
      color: '#e8eef5',
      fontSize: 14,
      fontWeight: 500,
    },
  }))
}

// Maps relationship id -> id of its inverse partner: imported from the routing
// lib (findInversePartnerIds) so the edge styling and the lane plan always agree
// on which relationships are inverse pairs.

function buildEdges(relationships, routing, selectRef) {
  if (!relationships || relationships.length === 0) return []
  return relationships.map((rel) => {
    const route = routing[rel.id] || {}
    const partnerId = route.inversePartnerId || null
    return {
      id: rel.id,
      source: rel.subject_id,
      target: rel.object_id,
      // Kept for semantics/debugging; the custom edge renders data.predicate
      // itself so the label can be positioned on its own lane.
      label: rel.predicate,
      type: 'lane',
      data: {
        predicate: rel.predicate,
        confidence: rel.confidence,
        title: edgeTitle(rel.predicate, rel.confidence),
        inversePartnerId: partnerId,
        // Deterministic lane routing (src/lib/edgeRouting.js).
        laneOffset: route.laneOffset || 0,
        labelT: route.labelT,
        curvature: route.curvature,
        labelSelected: false,
        // Stable indirection: the callback identity may change between renders
        // (App re-creates handlers), the ref never does.
        onSelect: (edgeId) => selectRef.current(edgeId),
      },
      style: {
        stroke: EDGE_COLOR,
        strokeWidth: EDGE_DEFAULT_WIDTH,
        ...(partnerId ? { strokeDasharray: '6 4' } : {}),
      },
    }
  })
}

function layoutNodes(nodes, edges) {
  if (!nodes || nodes.length === 0) return nodes || []
  // Single source of truth: geometry lives in src/lib/edgeRouting.js (the
  // verifier lays out the identical rects through the same function).
  const objects = nodes.map((node) => ({ id: node.id }))
  const relationships = edges.map((edge, index) => ({
    id: edge.id || `edge_${index}`,
    subject_id: edge.source,
    object_id: edge.target,
  }))
  const rects = layoutNodeRects(objects, relationships, dagre)
  return nodes.map((node) => ({
    ...node,
    position: { x: rects[node.id].x, y: rects[node.id].y },
  }))
}

// Border for one node. Priority: object > relationship endpoint > search match.
function nodeBorderStyle(nodeId, selectedObjectId, relEndpointIds, searchIds) {
  if (nodeId === selectedObjectId) return `3px solid ${COLOR_SELECTED}`
  if (relEndpointIds.includes(nodeId)) return `2.5px solid ${COLOR_REL_ENDPOINT}`
  if (searchIds.includes(nodeId)) return `2.5px solid ${COLOR_SEARCH}`
  return `2px solid ${COLOR_NODE_BORDER}`
}

function InnerSceneGraph({
  scene,
  selectedObjectId,
  selectedRelationshipId,
  searchObjectIds,
  highlightedObjectIds,
  displayLabels,
  locked,
  hiddenObjectIds,
  hiddenRelationshipIds,
  onShowAllVisibility,
  onObjectSelect,
  onRelationshipSelect,
  onPaneClick,
  onToggleLock,
}) {
  const objects = scene?.objects
  const relationships = scene?.relationships

  // --- graph visibility (frontend-only; the rule lives in the lib) --------
  const hiddenObjects = hiddenObjectIds || EMPTY_SET
  const hiddenRelationships = hiddenRelationshipIds || EMPTY_SET

  // Built once per scene: dagre layout for nodes, one edge per relationship.
  // Node positions after this point belong to the user (drag), so this memo is
  // the ONLY source of dagre layout and it never re-runs on selection changes.
  // The lane plan is derived from the FULL relationship list, so hiding an
  // element never re-routes the remaining edges.
  // Deterministic lane routing for the WHOLE scene (see src/lib/edgeRouting.js).
  const routing = useMemo(() => planEdgeRouting(relationships), [relationships])

  // Always-current selection callback for the clickable edge labels.
  const selectRef = useRef(onRelationshipSelect)
  useEffect(() => {
    selectRef.current = onRelationshipSelect
  }, [onRelationshipSelect])

  const sceneData = useMemo(
    () => ({
      nodes: layoutNodes(
        buildNodes(objects, displayLabels),
        buildEdges(relationships, routing, selectRef),
      ),
      edges: buildEdges(relationships, routing, selectRef),
    }),
    [objects, relationships, displayLabels, routing],
  )

  const [nodes, setNodes, onNodesChange] = useNodesState(sceneData.nodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(sceneData.edges)

  // What is actually drawn.  Positions stay in `nodes`/`edges` state, so hiding
  // and re-showing an element never loses a dragged position.
  const visibleNodeIdSet = useMemo(
    () => visibleNodeIds(objects, hiddenObjects),
    [objects, hiddenObjects],
  )
  const visibleEdgeIdSet = useMemo(
    () => visibleEdgeIds(relationships, hiddenObjects, hiddenRelationships),
    [relationships, hiddenObjects, hiddenRelationships],
  )
  const drawnNodes = useMemo(
    () => nodes.filter((node) => visibleNodeIdSet.has(node.id)),
    [nodes, visibleNodeIdSet],
  )
  const visibleEdges = useMemo(
    () => edges.filter((edge) => visibleEdgeIdSet.has(edge.id)),
    [edges, visibleEdgeIdSet],
  )

  // Node rectangles in flow coordinates (positions live in the node state, the
  // size is fixed) - the obstacles every label has to avoid.
  const nodeRects = useMemo(() => {
    const rects = {}
    for (const node of nodes) {
      if (!visibleNodeIdSet.has(node.id)) continue
      rects[node.id] = {
        x: node.position.x,
        y: node.position.y,
        w: NODE_WIDTH,
        h: NODE_HEIGHT,
      }
    }
    return rects
  }, [nodes, visibleNodeIdSet])

  // One deterministic pass for the whole graph: every label dodges the nodes
  // plus the labels placed before it (scene order), so two labels can never end
  // up on the same spot.  Recomputed while dragging, still fully deterministic.
  const labelPlacement = useMemo(
    () =>
      placeLabels({
        edges: visibleEdges.map((edge) => ({
          id: edge.id,
          source: edge.source,
          target: edge.target,
          predicate: edge.data?.predicate,
        })),
        plan: routing,
        nodeRects,
      }),
    [visibleEdges, routing, nodeRects],
  )

  const drawnEdges = useMemo(
    () =>
      visibleEdges.map((edge) => {
        const point = labelPlacement[edge.id]
        if (!point) return edge
        return { ...edge, data: { ...edge.data, labelPoint: { x: point.x0, y: point.y0 } } }
      }),
    [visibleEdges, labelPlacement],
  )

  const hiddenCount =
    (objects ? objects.length - visibleNodeIdSet.size : 0) +
    (relationships ? relationships.length - visibleEdgeIdSet.size : 0)

  const wrapperRef = useRef(null)
  const { fitView } = useReactFlow()

  const relEndpointIds = highlightedObjectIds || []
  const searchIds = searchObjectIds || []

  // Restyle existing nodes whenever selection or search changes.
  useEffect(() => {
    setNodes((current) =>
      current.map((node) => ({
        ...node,
        style: {
          ...node.style,
          border: nodeBorderStyle(node.id, selectedObjectId, relEndpointIds, searchIds),
        },
      })),
    )
  }, [selectedObjectId, relEndpointIds, searchIds, setNodes])

  // Restyle existing edges whenever the selected relationship changes.  The
  // label colour follows the same flag (the custom edge reads data.labelSelected).
  useEffect(() => {
    setEdges((current) =>
      current.map((edge) => {
        const selected = edge.id === selectedRelationshipId
        return {
          ...edge,
          style: {
            ...edge.style,
            stroke: selected ? EDGE_COLOR_SELECTED : EDGE_COLOR,
            strokeWidth: selected ? 3 : EDGE_DEFAULT_WIDTH,
          },
          data: { ...edge.data, labelSelected: selected },
        }
      }),
    )
  }, [selectedRelationshipId, setEdges])

  // Export the graph canvas only (no minimap, controls or attribution).
  const handleDownload = useCallback(async () => {
    if (!wrapperRef.current) return
    try {
      await fitView({ padding: 0.1 })
      await new Promise((resolve) => setTimeout(resolve, 80))
      const dataUrl = await toPng(wrapperRef.current, {
        cacheBust: true,
        backgroundColor: '#0f1620',
        filter: (node) => {
          const cls = node.classList
          if (!cls) return true
          return !(
            cls.contains('react-flow__minimap') ||
            cls.contains('react-flow__controls') ||
            cls.contains('react-flow__panel') ||
            cls.contains('react-flow__attribution')
          )
        },
      })
      const link = document.createElement('a')
      link.download = 'scene-graph.png'
      link.href = dataUrl
      link.click()
    } catch (err) {
      console.error('Failed to export graph PNG:', err)
    }
  }, [fitView])

  if (!objects || objects.length === 0) {
    return <div className="graph-empty">No objects detected in this image.</div>
  }

  const noRelationships = !relationships || relationships.length === 0

  return (
    <div className="graph-wrapper">
      <div className="graph-toolbar">
        <div className="graph-toolbar-group">
          <button
            type="button"
            className={`btn btn-secondary btn-sm lock-toggle${locked ? ' is-locked' : ''}`}
            onClick={() => onToggleLock(!locked)}
            aria-pressed={locked}
            title={
              locked
                ? 'Locked: click to make the nodes draggable again'
                : 'Unlocked: click to freeze the node positions'
            }
          >
            <span className="lock-icon" aria-hidden="true">
              {locked ? '\u{1F512}' : '\u{1F513}'}
            </span>
            {locked ? 'Locked' : 'Unlocked'}
          </button>
          {noRelationships && <span className="graph-note">No relationships predicted.</span>}
          {hiddenCount > 0 && (
            <span className="graph-note graph-note-hidden">
              Hidden in graph: {objects.length - visibleNodeIdSet.size} object
              {objects.length - visibleNodeIdSet.size === 1 ? '' : 's'},{' '}
              {relationships.length - visibleEdgeIdSet.size} relationship
              {relationships.length - visibleEdgeIdSet.size === 1 ? '' : 's'}
            </span>
          )}
          {hiddenCount > 0 && (
            <button
              type="button"
              className="btn btn-ghost btn-sm show-all-visibility"
              onClick={onShowAllVisibility}
              title="Show every object and relationship in the graph again"
            >
              Show all
            </button>
          )}
        </div>
        <button
          type="button"
          className="btn btn-secondary btn-sm"
          onClick={handleDownload}
          title="Download the scene graph as a PNG"
        >
          <span className="btn-icon" aria-hidden="true">
            {'\u2193'}
          </span>
          Download
        </button>
      </div>
      <div className="graph-container" ref={wrapperRef}>
        <ReactFlow
          nodes={drawnNodes}
          edges={drawnEdges}
          edgeTypes={EDGE_TYPES}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onNodeClick={(event, node) => onObjectSelect(node.id)}
          onEdgeClick={(event, edge) => onRelationshipSelect(edge.id)}
          onPaneClick={onPaneClick}
          onInit={(instance) => instance.fitView({ padding: 0.15 })}
          fitView
          minZoom={0.1}
          maxZoom={2.5}
          deleteKeyCode={null}
          proOptions={{ hideAttribution: true }}
          nodesConnectable={false}
          nodesDraggable={!locked}
        >
          <Background color="#1a2332" gap={20} />
          <Controls showInteractive={false} />
          <MiniMap
            nodeColor={COLOR_NODE_BORDER}
            maskColor="rgba(0, 0, 0, 0.25)"
            style={{ backgroundColor: '#101826' }}
            pannable
            zoomable
          />
        </ReactFlow>
      </div>
    </div>
  )
}

export default function SceneGraph(props) {
  return (
    <ReactFlowProvider>
      <InnerSceneGraph {...props} />
    </ReactFlowProvider>
  )
}
