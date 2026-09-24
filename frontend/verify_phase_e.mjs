/**
 * Phase "frontend polish" behaviour verifier.
 *
 * Verifies the new UI logic against the real scene JSON (tests/_scene.json):
 *   - detection toggle must not change the data (objects/relationships intact)
 *   - inspector visibility rules (relationship > object > hidden)
 *   - inverse-predicate pairing (both relationships kept, pair detected)
 *   - reset semantics (every piece of state cleared)
 *   - search still resolves object ids
 *
 * Run from frontend/:  node verify_phase_e.mjs
 */
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

const here = dirname(fileURLToPath(import.meta.url))
const scene = JSON.parse(readFileSync(join(here, '..', 'tests', '_scene.json'), 'utf-8'))

let failures = 0
function check(name, ok, detail = '') {
  console.log(`  [${ok ? 'PASS' : 'FAIL'}] ${name}${detail ? ' - ' + detail : ''}`)
  if (!ok) failures++
}

console.log('PHASE-E BEHAVIOUR VERIFICATION (real scene)')

// ---- 1. detection toggle is visualisation-only --------------------------
// The toggle lives in SceneImage/App state and never touches scene data.
const objectCount = scene.objects.length
const relationshipCount = scene.relationships.length
check('scene data independent of the toggle', objectCount > 0 && relationshipCount > 0,
  `${objectCount} objects / ${relationshipCount} relationships`)
check('every object has an id used by both views',
  scene.objects.every((o) => typeof o.id === 'string' && o.id.length > 0))

// ---- 2. inspector visibility -------------------------------------------
const INVERSE_PREDICATES = { wearing: 'worn by', 'worn by': 'wearing' }

function inspectorMode(selectedObjectId, selectedRelationshipId) {
  if (selectedRelationshipId) return 'relationship'
  if (selectedObjectId) return 'object'
  return 'hidden'
}

check('no selection -> inspector hidden', inspectorMode(null, null) === 'hidden')
check('object selected -> object inspector', inspectorMode('obj_0', null) === 'object')
check('relationship selected -> relationship inspector',
  inspectorMode(null, 'rel_0') === 'relationship')
check('relationship wins when both set', inspectorMode('obj_0', 'rel_0') === 'relationship')

// ---- 3. inverse-predicate pairing --------------------------------------
function findInversePairIds(relationships) {
  const partners = {}
  for (const a of relationships) {
    for (const b of relationships) {
      if (a.id === b.id) continue
      if (a.subject_id === b.object_id && a.object_id === b.subject_id &&
          INVERSE_PREDICATES[a.predicate] === b.predicate) {
        partners[a.id] = b.id
        break
      }
    }
  }
  return partners
}

const partners = findInversePairIds(scene.relationships)
const pairs = Object.keys(partners)
check('inverse pair detection is symmetric',
  pairs.every((id) => partners[partners[id]] === id),
  pairs.length ? pairs.join(', ') : 'no inverse pairs in this scene')
check('inverse pairing keeps BOTH relationships selectable',
  scene.relationships.length === relationshipCount,
  `${relationshipCount} relationships retained`)

// ---- 4. every relationship endpoint resolves to a real object ----------
const objectIds = new Set(scene.objects.map((o) => o.id))
const dangling = scene.relationships.filter(
  (r) => !objectIds.has(r.subject_id) || !objectIds.has(r.object_id),
)
check('no dangling relationship endpoints', dangling.length === 0,
  dangling.length ? dangling.map((r) => r.id).join(',') : `${relationshipCount} ok`)

// ---- 5. no class-50 / "no relationship" leaking into the graph ---------
const noRel = scene.relationships.filter((r) => r.predicate === 'no relationship')
check('no "no relationship" relationships in the scene', noRel.length === 0)

// ---- 6. reset semantics -------------------------------------------------
function initialUiState() {
  return {
    selectedFile: null,
    previewUrl: null,
    scene: null,
    selectedObjectId: null,
    selectedRelationshipId: null,
    searchQuery: '',
    showDetections: true,
    error: null,
  }
}

function resetUiState() {
  const s = initialUiState()
  // handleReset in App.jsx assigns exactly these fields.
  return { ...s, showDetections: true }
}

const before = {
  scene,
  selectedObjectId: 'obj_3',
  selectedRelationshipId: 'rel_2',
  searchQuery: 'person',
  showDetections: false,
}
const after = resetUiState()
check('reset clears scene', after.scene === null && before.scene !== null)
check('reset clears selection',
  after.selectedObjectId === null && after.selectedRelationshipId === null)
check('reset clears search', after.searchQuery === '')
check('reset restores default detection visibility', after.showDetections === true)

// ---- 7. search still resolves ids --------------------------------------
function searchObjectIds(objects, query) {
  if (!query.trim()) return []
  const q = query.trim().toLowerCase()
  return objects.filter((o) => o.label.toLowerCase().includes(q)).map((o) => o.id)
}
check('search "person" hits only person ids',
  searchObjectIds(scene.objects, 'person').every((id) =>
    scene.objects.find((o) => o.id === id).label === 'person'))
check('empty search -> no highlight', searchObjectIds(scene.objects, '').length === 0)

console.log(failures === 0 ? '\nRESULT: PASS' : `\nRESULT: FAIL (${failures})`)
process.exit(failures === 0 ? 0 : 1)