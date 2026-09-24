/**
 * Inspector - minimal details panel for the currently selected object or
 * relationship.
 *
 * Selection priority:
 *   selectedRelationshipId  -> Relationship Details
 *   selectedObjectId        -> Object Details
 *   neither                 -> placeholder
 *
 * Reads directly from the single scene source of truth.  Resolves subject /
 * object labels via their IDs (duplicate labels stay separate), and shows the
 * DISPLAY label ("person 2") alongside the internal id ("obj_1") so two objects
 * with the same label can never be confused.  The display label is derived
 * from the scene in App; nothing here mutates the scene.
 *
 * GRAPH VISIBILITY: each view carries a "Show in graph" switch.  Switching an
 * element off only stops the GRAPH from drawing it - the scene, the backend
 * response, the image detections and the search index are untouched.  The
 * switch reflects hiddenObjectIds / hiddenRelationshipIds, which App owns (see
 * src/lib/graphVisibility.js for the rules, including the fact that hiding an
 * object also hides every relationship incident to it).
 */
const EMPTY_SET = new Set()

export default function Inspector({
  scene,
  selectedObjectId,
  selectedRelationshipId,
  displayLabels,
  hiddenObjectIds,
  hiddenRelationshipIds,
  onToggleObjectVisibility,
  onToggleRelationshipVisibility,
}) {
  const labels = displayLabels || {}
  const hiddenObjects = hiddenObjectIds || EMPTY_SET
  const hiddenRelationships = hiddenRelationshipIds || EMPTY_SET
  // Presentation-only label lookup; falls back to the raw model label.
  const labelFor = (obj) => labels[obj.id] || obj.label

  if (!scene) {
    return (
      <div className="inspector">
        <h2>Inspector</h2>
        <p className="inspector-empty">Analyze an image first.</p>
      </div>
    )
  }

  // Relationship details take priority.
  if (selectedRelationshipId) {
    const rel = scene.relationships.find((r) => r.id === selectedRelationshipId)
    if (!rel) {
      return (
        <div className="inspector">
          <h2>Inspector</h2>
          <p className="inspector-empty">Select an object or relationship to inspect.</p>
        </div>
      )
    }
    const subject = scene.objects.find((o) => o.id === rel.subject_id)
    const object = scene.objects.find((o) => o.id === rel.object_id)
    return (
      <div className="inspector">
        <h2>Relationship Details</h2>
        <dl className="inspector-list">
          <dt>Subject</dt>
          <dd>{subject ? labelFor(subject) : rel.subject_id}</dd>

          <dt>Predicate</dt>
          <dd>{rel.predicate}</dd>

          <dt>Object</dt>
          <dd>{object ? labelFor(object) : rel.object_id}</dd>

          <dt>Confidence</dt>
          <dd>{formatConfidence(rel.confidence)}</dd>
        </dl>

        <VisibilityControl
          id="relationship-visibility-toggle"
          visible={!hiddenRelationships.has(rel.id)}
          onChange={() => onToggleRelationshipVisibility(rel.id)}
          visibleHint="Drawn in the graph"
          hiddenHint="This relationship is hidden from the graph. Its objects stay."
        />
      </div>
    )
  }

  // Object details.
  if (selectedObjectId) {
    const obj = scene.objects.find((o) => o.id === selectedObjectId)
    if (!obj) {
      return (
        <div className="inspector">
          <h2>Inspector</h2>
          <p className="inspector-empty">Select an object or relationship to inspect.</p>
        </div>
      )
    }
    const outgoing = scene.relationships.filter((r) => r.subject_id === obj.id)
    const incoming = scene.relationships.filter((r) => r.object_id === obj.id)

    return (
      <div className="inspector">
        <h2>Object Details</h2>
        <dl className="inspector-list">
          <dt>Label</dt>
          <dd>{labelFor(obj)}</dd>

          <dt>ID</dt>
          <dd>{obj.id}</dd>

          <dt>Confidence</dt>
          <dd>{formatConfidence(obj.confidence)}</dd>

          <dt>Bounding Box</dt>
          <dd>
            <span className="bbox-values">
              cx: {formatNum(obj.bbox.cx)}, cy: {formatNum(obj.bbox.cy)},{' '}
              w: {formatNum(obj.bbox.width)}, h: {formatNum(obj.bbox.height)}
            </span>
          </dd>
        </dl>

        <VisibilityControl
          id="object-visibility-toggle"
          visible={!hiddenObjects.has(obj.id)}
          onChange={() => onToggleObjectVisibility(obj.id)}
          visibleHint="Node and its relationships are drawn in the graph"
          hiddenHint="Hidden from the graph, together with every relationship touching it. The image detection and search are unaffected."
        />

        <div className="inspector-relationships">
          <h3>Relationships</h3>
          {outgoing.length === 0 && incoming.length === 0 && (
            <p className="inspector-empty">No relationships.</p>
          )}
          {outgoing.length > 0 && (
            <div>
              <h4>Outgoing</h4>
              <ul>
                {outgoing.map((r) => {
                  const target = scene.objects.find((o) => o.id === r.object_id)
                  return (
                    <li key={r.id}>
                      {r.predicate} &rarr; {target ? labelFor(target) : r.object_id}{' '}
                      <span className="rel-conf">({formatConfidence(r.confidence)})</span>
                    </li>
                  )
                })}
              </ul>
            </div>
          )}
          {incoming.length > 0 && (
            <div>
              <h4>Incoming</h4>
              <ul>
                {incoming.map((r) => {
                  const source = scene.objects.find((o) => o.id === r.subject_id)
                  return (
                    <li key={r.id}>
                      {source ? labelFor(source) : r.subject_id} &rarr; {r.predicate}{' '}
                      <span className="rel-conf">({formatConfidence(r.confidence)})</span>
                    </li>
                  )
                })}
              </ul>
            </div>
          )}
        </div>
      </div>
    )
  }

  return (
    <div className="inspector">
      <h2>Inspector</h2>
      <p className="inspector-empty">Select an object or relationship to inspect.</p>
    </div>
  )
}

function formatConfidence(value) {
  if (value == null) return '-'
  return `${(value * 100).toFixed(2)}%`
}

function formatNum(value) {
  if (value == null) return '-'
  return Number.isInteger(value) ? value : value.toFixed(1)
}

/**
 * "Show in graph" switch.
 *
 * Presentation only: it reflects one id's membership in the hidden set and
 * asks App to toggle it.  Uses the same switch styling as the detection toggle
 * in the image panel, so the two visibility controls read as one family.
 */
function VisibilityControl({ id, visible, onChange, visibleHint, hiddenHint }) {
  return (
    <div className={`visibility-row${visible ? '' : ' is-hidden-element'}`}>
      <span className="visibility-label">
        <span className="visibility-eye" aria-hidden="true">
          {'\u{1F441}'}
        </span>
        Show in graph
      </span>
      <label className="toggle toggle-compact" htmlFor={id}>
        <input
          id={id}
          className="toggle-input"
          type="checkbox"
          checked={visible}
          onChange={onChange}
          data-testid={id}
        />
        <span className="toggle-track" aria-hidden="true" />
      </label>
      <p className="visibility-hint">{visible ? visibleHint : hiddenHint}</p>
    </div>
  )
}