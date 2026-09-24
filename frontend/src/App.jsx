import { useState, useRef, useMemo } from 'react'
import { analyzeImage } from './services/api.js'
import SceneImage from './components/SceneImage.jsx'
import SceneGraph from './components/SceneGraph.jsx'
import Inspector from './components/Inspector.jsx'
import {
  resolveSelectionAfterVisibilityChange,
  toggleHiddenId,
} from './lib/graphVisibility.js'

/**
 * App - owns every piece of frontend state:
 *
 *   upload / analyze flow, the scene returned by FastAPI, the shared selection
 *   ids (image <-> graph), the object search query and the detection-visibility
 *   toggle.
 *
 * The object id ("obj_0") is the single bridge between the image view, the
 * graph view and the inspector.  Both views read the SAME scene object - there
 * is never a second copy of the data.
 */
export default function App() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [scene, setScene] = useState(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [error, setError] = useState(null)

  // Shared selection state, owned here so both views see the same ids.
  const [selectedObjectId, setSelectedObjectId] = useState(null)
  const [selectedRelationshipId, setSelectedRelationshipId] = useState(null)

  // Object search by label (case-insensitive substring). Empty = no search.
  const [searchQuery, setSearchQuery] = useState('')

  // VISUALISATION-ONLY switch: sets the DEFAULT visibility of the detection
  // overlays.  An active selection still shows its own highlight - see
  // visibleObjectIds below.  It never touches the scene data.
  const [showDetections, setShowDetections] = useState(true)

  // Graph lock (frontend-only UI state).  false = nodes are draggable, which is
  // the default.  Kept here so it survives a graph remount.
  const [graphLocked, setGraphLocked] = useState(false)

  // GRAPH VISIBILITY (frontend-only).  These ids only control what the GRAPH
  // draws - nothing is removed from the scene, the backend response, the image
  // detections or the search index.  Two independent sets, so hiding an object
  // and hiding a relationship never overwrite each other's choice.  The rule
  // that combines them lives in src/lib/graphVisibility.js.
  const [hiddenObjectIds, setHiddenObjectIds] = useState(() => new Set())
  const [hiddenRelationshipIds, setHiddenRelationshipIds] = useState(() => new Set())

  // Bumped on every successful analyze so the explorer remounts with a fresh
  // dagre layout for the new scene.
  const [sceneKey, setSceneKey] = useState(0)

  const fileInputRef = useRef(null)

  // Object ids matching the current query. Drives BOTH the image and the graph.
  const searchObjectIds = useMemo(() => {
    if (!searchQuery.trim() || !scene?.objects) return []
    const q = searchQuery.trim().toLowerCase()
    return scene.objects.filter((o) => o.label.toLowerCase().includes(q)).map((o) => o.id)
  }, [searchQuery, scene])

  // Endpoint ids of the selected relationship - highlighted in BOTH views.
  const relEndpointIds = useMemo(() => {
    if (!selectedRelationshipId || !scene?.relationships) return []
    const rel = scene.relationships.find((r) => r.id === selectedRelationshipId)
    return rel ? [rel.subject_id, rel.object_id] : []
  }, [selectedRelationshipId, scene])

  // Display-only labels: "person 1" / "person 2" when a scene contains the same
  // label more than once, plain "car" when it does not.  Numbering follows
  // scene.objects order, so it is deterministic for a given scene.  The scene
  // itself is NEVER modified: object.id, object.label and the relationship
  // subject_id/object_id keep their original values - this map is presentation.
  const displayLabels = useMemo(() => {
    const labels = {}
    if (!scene?.objects) return labels
    const totals = {}
    for (const obj of scene.objects) {
      totals[obj.label] = (totals[obj.label] || 0) + 1
    }
    const seen = {}
    for (const obj of scene.objects) {
      if (totals[obj.label] > 1) {
        seen[obj.label] = (seen[obj.label] || 0) + 1
        labels[obj.id] = `${obj.label} ${seen[obj.label]}`
      } else {
        labels[obj.id] = obj.label
      }
    }
    return labels
  }, [scene])

  // Which bounding boxes the image view paints.  The toggle decides the
  // DEFAULT; a selection always wins, so highlighting can never be suppressed:
  //
  //   detections ON                          -> all objects
  //   detections OFF + selected relationship -> the two endpoints only
  //   detections OFF + selected object       -> that object only
  //   detections OFF + nothing selected      -> nothing
  //
  // scene.objects is never filtered or mutated - this is a visibility list.
  const visibleObjectIds = useMemo(() => {
    if (!scene?.objects) return []
    if (showDetections) return scene.objects.map((obj) => obj.id)
    if (selectedRelationshipId) return relEndpointIds
    if (selectedObjectId) return [selectedObjectId]
    return []
  }, [scene, showDetections, selectedRelationshipId, selectedObjectId, relEndpointIds])

  const hasSelection = Boolean(selectedObjectId || selectedRelationshipId)

  // --- graph visibility handlers -----------------------------------------
  // Every visibility change ends with the same selection cleanup, so the
  // inspector can never point at something the graph no longer draws.
  function applyVisibility(nextHiddenObjects, nextHiddenRelationships) {
    setHiddenObjectIds(nextHiddenObjects)
    setHiddenRelationshipIds(nextHiddenRelationships)
    const nextSelection = resolveSelectionAfterVisibilityChange({
      scene,
      selectedObjectId,
      selectedRelationshipId,
      hiddenObjectIds: nextHiddenObjects,
      hiddenRelationshipIds: nextHiddenRelationships,
    })
    setSelectedObjectId(nextSelection.selectedObjectId)
    setSelectedRelationshipId(nextSelection.selectedRelationshipId)
  }

  // Hiding an object also hides every relationship touching it (the rule lives
  // in the lib).  Re-showing it restores exactly the relationships that were
  // not hidden individually.
  function handleToggleObjectVisibility(objectId) {
    applyVisibility(toggleHiddenId(hiddenObjectIds, objectId), hiddenRelationshipIds)
  }

  // Hiding a relationship removes only that edge; its endpoints stay.
  function handleToggleRelationshipVisibility(relationshipId) {
    applyVisibility(hiddenObjectIds, toggleHiddenId(hiddenRelationshipIds, relationshipId))
  }

  function handleShowAllVisibility() {
    applyVisibility(new Set(), new Set())
  }

  function handleFileChange(event) {
    const file = event.target.files[0]
    if (!file) return

    if (!file.type.startsWith('image/')) {
      setError('Please select an image file (JPEG, PNG or WEBP).')
      return
    }

    setSelectedFile(file)
    setScene(null)
    setSelectedObjectId(null)
    setSelectedRelationshipId(null)
    setSearchQuery('')
    setHiddenObjectIds(new Set())
    setHiddenRelationshipIds(new Set())
    setError(null)

    const reader = new FileReader()
    reader.onload = (e) => setPreviewUrl(e.target.result)
    reader.readAsDataURL(file)
  }

  async function handleAnalyze() {
    if (!selectedFile) return

    setIsAnalyzing(true)
    setError(null)
    setScene(null)
    setSelectedObjectId(null)
    setSelectedRelationshipId(null)
    // A new analysis starts from a clean, fully visible graph: hidden ids must
    // never leak from the previous scene into the new one.
    setHiddenObjectIds(new Set())
    setHiddenRelationshipIds(new Set())

    try {
      const result = await analyzeImage(selectedFile)
      setScene(result)
      setSceneKey((key) => key + 1)
    } catch (err) {
      setError(err.message)
    } finally {
      setIsAnalyzing(false)
    }
  }

  // Reset returns the whole app to its initial state - no stale graph,
  // inspector, search or selection state survives.
  function handleReset() {
    setSelectedFile(null)
    setPreviewUrl(null)
    setScene(null)
    setSelectedObjectId(null)
    setSelectedRelationshipId(null)
    setSearchQuery('')
    setShowDetections(true)
    setGraphLocked(false)
    setHiddenObjectIds(new Set())
    setHiddenRelationshipIds(new Set())
    setError(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  // Image -> graph sync: clicking a bounding box selects the object.
  function handleObjectSelect(objectId) {
    setSelectedObjectId(objectId)
    setSelectedRelationshipId(null)
  }

  // Graph -> image sync: clicking an edge selects the relationship and clears
  // the object selection so the inspector priority stays unambiguous.
  function handleRelationshipSelect(relationshipId) {
    setSelectedRelationshipId(relationshipId)
    setSelectedObjectId(null)
  }

  // Clicking empty graph canvas clears the selection (and hides the inspector).
  function handlePaneClick() {
    setSelectedObjectId(null)
    setSelectedRelationshipId(null)
  }

  // Header file actions only make sense once there is a file or a scene: the
  // landing hero owns the "Choose Image" call to action.
  const showFileActions = Boolean(selectedFile || scene)

  return (
    <div className={`app${scene ? ' is-explorer' : ''}`}>
      {/* Always mounted, so the file picker is reachable from every state
          (landing hero, preview and explorer). */}
      <input
        ref={fileInputRef}
        id="file-input"
        type="file"
        accept="image/*"
        onChange={handleFileChange}
        hidden
      />

      <header className="app-header">
        <div className="app-brand">
          <span className="brand-mark" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none">
              <path d="M12 5.5v4.2M12 9.7 7.2 12.9M12 9.7l4.8 3.2M7.2 12.9v3.5" />
              <circle cx="12" cy="4.4" r="2.1" />
              <circle cx="7.2" cy="12.4" r="1.9" />
              <circle cx="16.8" cy="12.4" r="1.9" />
              <circle cx="7.2" cy="17.8" r="1.9" />
            </svg>
          </span>
          <div className="brand-text">
            <h1>Scene Graph Explorer</h1>
            <p className="subtitle">
              Visual scene understanding through object detection and relationship prediction
            </p>
          </div>
        </div>

        {/* File actions appear once there is something to act on; the landing
            hero owns the single "Choose Image" call to action. */}
        {showFileActions && (
          <div className="header-actions">
            <button
              type="button"
              className="btn btn-secondary"
              onClick={() => fileInputRef.current?.click()}
            >
              Choose Image
            </button>
            <button
              type="button"
              className="btn btn-primary"
              onClick={handleAnalyze}
              disabled={!selectedFile || isAnalyzing}
            >
              {isAnalyzing ? 'Analyzing...' : 'Analyze'}
            </button>
            <button
              type="button"
              className="btn btn-ghost"
              onClick={handleReset}
              disabled={!selectedFile && !scene}
            >
              Reset
            </button>
          </div>
        )}
      </header>

      {error && <div className="error-box">{error}</div>}

      {!selectedFile && !scene && !isAnalyzing && (
        <section className="empty-state">
          <div className="hero-icon" aria-hidden="true">
            <svg viewBox="0 0 130 92" fill="none">
              <path d="M65 20 30 50M65 20l35 30M30 50h70M30 50l35 24M100 50l-35 24" />
              <circle cx="65" cy="20" r="9" />
              <circle cx="30" cy="50" r="7.5" />
              <circle cx="100" cy="50" r="7.5" />
              <circle cx="65" cy="74" r="7.5" />
            </svg>
          </div>
          <h2>Upload an image to explore its scene graph</h2>
          <p className="hero-text">
            The detector finds the objects, the relationship model predicts how they relate,
            and the image and graph stay linked - click a box, a node or an edge to inspect it.
          </p>
          <button
            type="button"
            className="btn btn-primary btn-lg"
            onClick={() => fileInputRef.current?.click()}
          >
            Choose Image
          </button>
          <p className="hero-hint">JPEG, PNG or WEBP</p>
        </section>
      )}

      {isAnalyzing && (
        <section className="loading-state">
          <div className="loading-card">
            <div className="spinner" />
            <p className="loading-text">Analyzing image...</p>
            {selectedFile && <p className="loading-file">{selectedFile.name}</p>}
          </div>
        </section>
      )}

      {previewUrl && !scene && !isAnalyzing && (
        <section className="preview-section">
          <img src={previewUrl} alt="Selected" className="preview-image" />
          {selectedFile && <p className="file-name">{selectedFile.name}</p>}
          <button type="button" className="btn btn-primary" onClick={handleAnalyze}>
            Run analysis
          </button>
        </section>
      )}

      {scene && (
        <section className="results-section">
          <div className="results-grid">
            <div className="result-stat">
              <span className="stat-num">{String(scene.objects.length).padStart(2, '0')}</span>
              <span className="stat-label">OBJECTS</span>
            </div>
            <div className="result-stat">
              <span className="stat-num">
                {String(scene.relationships.length).padStart(2, '0')}
              </span>
              <span className="stat-label">RELATIONSHIPS</span>
            </div>
          </div>
        </section>
      )}

      {scene && (
        <section className="explorer" key={sceneKey}>
          <div className="explorer-panel explorer-panel-image">
            <div className="panel-header">
              <h2>Image Viewer</h2>
              <label className="toggle" htmlFor="detections-toggle">
                <input
                  id="detections-toggle"
                  className="toggle-input"
                  type="checkbox"
                  checked={showDetections}
                  onChange={(e) => setShowDetections(e.target.checked)}
                />
                <span className="toggle-track" aria-hidden="true" />
                <span className="toggle-text">Show object detections</span>
              </label>
            </div>
            <SceneImage
              imageSrc={previewUrl}
              width={scene.image.width}
              height={scene.image.height}
              objects={scene.objects}
              displayLabels={displayLabels}
              visibleObjectIds={visibleObjectIds}
              selectedObjectId={selectedObjectId}
              highlightedObjectIds={relEndpointIds}
              searchObjectIds={searchObjectIds}
              showDetections={showDetections}
              onObjectClick={handleObjectSelect}
            />
          </div>

          <div className="explorer-panel explorer-panel-graph">
            <div className="panel-header">
              <h2>Scene Graph</h2>
            </div>
            <SceneGraph
              scene={scene}
              selectedObjectId={selectedObjectId}
              selectedRelationshipId={selectedRelationshipId}
              highlightedObjectIds={relEndpointIds}
              searchObjectIds={searchObjectIds}
              displayLabels={displayLabels}
              locked={graphLocked}
              hiddenObjectIds={hiddenObjectIds}
              hiddenRelationshipIds={hiddenRelationshipIds}
              onShowAllVisibility={handleShowAllVisibility}
              onToggleLock={setGraphLocked}
              onObjectSelect={handleObjectSelect}
              onRelationshipSelect={handleRelationshipSelect}
              onPaneClick={handlePaneClick}
            />
          </div>
        </section>
      )}

      {scene && (
        <section className="search-section">
          <div className="search-field">
            <span className="search-icon" aria-hidden="true">
              {'\u2315'}
            </span>
            <label className="sr-only" htmlFor="search-input">
              Search objects
            </label>
            <input
              id="search-input"
              type="text"
              placeholder="Search objects (e.g. person)"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>
          {searchQuery.trim() && (
            <span className="search-count">
              {searchObjectIds.length} match{searchObjectIds.length === 1 ? '' : 'es'}
            </span>
          )}
        </section>
      )}

      {/* The inspector only exists while something is selected. */}
      {scene && !hasSelection && (
        <p className="hint-line">Click an object or a relationship to inspect it.</p>
      )}

      {scene && hasSelection && (
        <section className="inspector-section">
          <Inspector
            scene={scene}
            selectedObjectId={selectedObjectId}
            selectedRelationshipId={selectedRelationshipId}
            displayLabels={displayLabels}
            hiddenObjectIds={hiddenObjectIds}
            hiddenRelationshipIds={hiddenRelationshipIds}
            onToggleObjectVisibility={handleToggleObjectVisibility}
            onToggleRelationshipVisibility={handleToggleRelationshipVisibility}
          />
        </section>
      )}
    </div>
  )
}
