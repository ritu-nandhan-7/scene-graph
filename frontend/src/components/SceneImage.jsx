/**
 * SceneImage - analyzed image viewer with bounding-box overlays.
 *
 * Bounding boxes are positioned as PERCENTAGES of the displayed image (never
 * fixed viewport pixels), so they stay aligned with the image at any
 * responsive size:
 *
 *     left%   = (cx - w/2) / imageWidth  * 100
 *     top%    = (cy - h/2) / imageHeight * 100
 *     width%  =  w / imageWidth   * 100
 *     height% =  h / imageHeight  * 100
 *
 * (cx, cy, w, h) are exactly what the backend returns: ORIGINAL image pixels
 * in centre format.  They are never reinterpreted or rescaled.
 *
 * The wrapper shrink-wraps the <img> element, so the percentages above always
 * refer to the RENDERED image rectangle - that is what keeps the boxes aligned
 * when the panel is resized or the image is height-capped by the matched panel
 * layout.
 *
 * VISIBILITY (see VISIBLE props below)
 * -----------------------------------
 * Which boxes are drawn is decided by the caller (`visibleObjectIds`), not by
 * this component.  The rules are:
 *
 *     detections ON                          -> every object
 *     detections OFF + selected object       -> that object only
 *     detections OFF + selected relationship -> the two endpoints only
 *     detections OFF + nothing selected      -> nothing
 *
 * So the toggle controls the DEFAULT visibility of the detections, and it can
 * never suppress a selection-driven highlight.  The scene data itself is never
 * touched: this component only decides what to paint.
 *
 * Object identity is ALWAYS scene.objects[i].id (e.g. "obj_0") and never the
 * label alone, because duplicate labels (person x2, tree x2) are separate
 * objects.  `displayLabels` supplies the human-readable label ("person 2")
 * derived from the id; it is presentation only.
 */
export default function SceneImage({
  imageSrc,
  width,
  height,
  objects,
  displayLabels,
  visibleObjectIds,
  selectedObjectId,
  highlightedObjectIds,
  searchObjectIds,
  showDetections,
  onObjectClick,
}) {
  const list = objects || []
  const labels = displayLabels || {}
  const highlighted = highlightedObjectIds || []
  const searched = searchObjectIds || []

  // Only the objects the caller marked visible are painted.
  const visibleSet = new Set(visibleObjectIds || [])
  const shown = list.filter((obj) => visibleSet.has(obj.id))

  // Percentage box for one object, aligned to the rendered image.
  function boxStyle(obj) {
    const b = obj.bbox
    return {
      left: `${((b.cx - b.width / 2) / width) * 100}%`,
      top: `${((b.cy - b.height / 2) / height) * 100}%`,
      width: `${(b.width / width) * 100}%`,
      height: `${(b.height / height) * 100}%`,
    }
  }

  // Priority: selected (red) > relationship endpoint (orange) > search (teal).
  function boxClass(obj) {
    if (obj.id === selectedObjectId) return 'bbox bbox-selected'
    if (highlighted.includes(obj.id)) return 'bbox bbox-highlighted'
    if (searched.includes(obj.id)) return 'bbox bbox-searched'
    return 'bbox'
  }

  function displayLabel(obj) {
    return labels[obj.id] || obj.label
  }

  const total = list.length
  const shownCount = shown.length
  let caption = `${total} object${total === 1 ? '' : 's'}`
  if (!showDetections) {
    caption +=
      shownCount === 0
        ? ' - detections hidden'
        : ` - showing ${shownCount} selected`
  }

  return (
    <div className="scene-image">
      <div className="scene-image-body">
        {/* The wrapper carries the image's aspect ratio so it can shrink to fit
            the fixed panel height while staying exactly the size of the painted
            image - which is what keeps the percentage boxes aligned. */}
        <div className="scene-image-wrap" style={{ aspectRatio: `${width} / ${height}` }}>
          <img
            src={imageSrc}
            alt="Analyzed scene"
            className="scene-image-img"
            draggable={false}
          />
          <div className="bbox-layer">
            {shown.map((obj) => (
              <button
                key={obj.id}
                type="button"
                className={boxClass(obj)}
                style={boxStyle(obj)}
                data-object-id={obj.id}
                data-display-label={displayLabel(obj)}
                title={`${displayLabel(obj)} (${obj.id})`}
                onMouseDown={(e) => e.preventDefault()}
                onClick={(e) => {
                  e.stopPropagation()
                  onObjectClick(obj.id)
                }}
              >
                <span className="bbox-label">
                  {displayLabel(obj)}
                  {obj.confidence != null ? ` ${(obj.confidence * 100).toFixed(0)}%` : ''}
                </span>
              </button>
            ))}
          </div>
        </div>
      </div>
      <p className="scene-image-meta">{caption}</p>
    </div>
  )
}
