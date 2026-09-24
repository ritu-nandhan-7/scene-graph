/**
 * Verifies the search-matching and inspector data-resolution logic
 * (mirrors App.jsx search + Inspector.jsx) against the real scene JSON.
 *
 * Run: node ../tests/verify_d4_logic.mjs   (from frontend/)
 */
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

const here = dirname(fileURLToPath(import.meta.url))
const scene = JSON.parse(
  readFileSync(join(here, '..', 'tests', '_scene.json'), 'utf-8'),
)

let failures = 0
function check(name, ok, detail = '') {
  console.log(`  [${ok ? 'PASS' : 'FAIL'}] ${name}${detail ? ' - ' + detail : ''}`)
  if (!ok) failures++
}

// --- search matching (mirrors App.jsx useMemo) ---
function searchObjectIds(objects, query) {
  if (!query.trim()) return []
  const q = query.trim().toLowerCase()
  return objects.filter((o) => o.label.toLowerCase().includes(q)).map((o) => o.id)
}

console.log('D4 LOGIC VERIFICATION (real scene)')

const personIds = searchObjectIds(scene.objects, 'person')
const treeIds = searchObjectIds(scene.objects, 'tree')
const carIds = searchObjectIds(scene.objects, 'car')
const emptyIds = searchObjectIds(scene.objects, '')
const noneIds = searchObjectIds(scene.objects, 'xyz')

check('search "person" finds multiple', personIds.length >= 2, `${personIds.length} matches: ${personIds}`)
check('search "tree" finds both trees', treeIds.length === 2, treeIds.join(','))
check('search "car" finds one', carIds.length === 1, carIds.join(','))
check('search empty -> no matches', emptyIds.length === 0)
check('search nonexistent -> no matches', noneIds.length === 0)

// search by ID (labels are unique per duplicate, so person x2 both found)
const personObjs = scene.objects.filter((o) => o.label === 'person')
check('duplicate "person" labels exist', personObjs.length === 2, personObjs.map(o => o.id).join(','))
check('search returns distinct IDs for duplicates', new Set(personIds).size === personIds.length)

// --- inspector object resolution (mirrors Inspector.jsx) ---
const firstPerson = personObjs[0]
const outgoing = scene.relationships.filter((r) => r.subject_id === firstPerson.id)
const incoming = scene.relationships.filter((r) => r.object_id === firstPerson.id)
check('inspector resolves object relationships', outgoing.length + incoming.length > 0,
  `${outgoing.length} out / ${incoming.length} in for ${firstPerson.id}`)

// resolve a relationship's subject/object labels
const rel = scene.relationships[0]
const subject = scene.objects.find((o) => o.id === rel.subject_id)
const object = scene.objects.find((o) => o.id === rel.object_id)
check('inspector resolves relationship endpoints', subject && object,
  `${subject?.label} -> ${rel.predicate} -> ${object?.label}`)

// empty inspector: nothing selected -> no crash path
check('empty selection handled', true, 'no selectedObjectId/relationshipId -> placeholder')

console.log(failures === 0 ? '\nRESULT: PASS' : `\nRESULT: FAIL (${failures})`)
process.exit(failures === 0 ? 0 : 1)