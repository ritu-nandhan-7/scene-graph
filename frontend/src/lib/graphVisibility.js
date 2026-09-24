/**
 * Graph visibility - frontend ONLY, pure functions.
 *
 * "Hidden" NEVER means deleted.  Switching an element off hides it from the
 * GRAPH render; the backend response, scene.objects, scene.relationships, the
 * image detections and the search index are all untouched.
 *
 * Two independent sets of ids are kept, so the user's two kinds of choice do
 * not overwrite each other:
 *
 *     hiddenObjectIds        object nodes the user switched off
 *     hiddenRelationshipIds  relationships the user switched off
 *
 * THE ONE RULE that decides whether an edge may be drawn:
 *
 *     edgeVisible = !relationshipHidden
 *                && !subjectHidden
 *                && !objectHidden
 *
 * It lives here (and only here) so the graph cannot accidentally render an
 * edge whose endpoint is hidden.  Consequences, all intentional:
 *
 *   - hiding an object removes every relationship touching it, even though
 *     those relationship ids were never added to hiddenRelationshipIds,
 *   - re-showing the object brings back exactly those relationships that were
 *     not hidden individually (the two sets stay independent),
 *   - hiding a relationship removes only that edge; its endpoints stay.
 *
 * Nothing here mutates the scene and nothing here renders.
 */

/** Immutable toggle: returns a NEW Set (React state friendly). */
export function toggleHiddenId(hiddenIds, id) {
  const next = new Set(hiddenIds)
  if (next.has(id)) next.delete(id)
  else next.add(id)
  return next
}

export function isObjectGraphVisible(objectId, hiddenObjectIds) {
  return !hiddenObjectIds.has(objectId)
}

/** The single edge-visibility rule described above. */
export function isRelationshipGraphVisible(rel, hiddenObjectIds, hiddenRelationshipIds) {
  if (hiddenRelationshipIds.has(rel.id)) return false
  if (hiddenObjectIds.has(rel.subject_id)) return false
  if (hiddenObjectIds.has(rel.object_id)) return false
  return true
}

/** Ids of the object nodes that may be drawn. */
export function visibleNodeIds(objects, hiddenObjectIds) {
  const ids = new Set()
  if (!objects) return ids
  for (const obj of objects) {
    if (isObjectGraphVisible(obj.id, hiddenObjectIds)) ids.add(obj.id)
  }
  return ids
}

/** Ids of the relationships that may be drawn. */
export function visibleEdgeIds(relationships, hiddenObjectIds, hiddenRelationshipIds) {
  const ids = new Set()
  if (!relationships) return ids
  for (const rel of relationships) {
    if (isRelationshipGraphVisible(rel, hiddenObjectIds, hiddenRelationshipIds)) {
      ids.add(rel.id)
    }
  }
  return ids
}

/** Counts for the graph toolbar summary (all derived, never stored). */
export function graphVisibilityCounts(scene, hiddenObjectIds, hiddenRelationshipIds) {
  const objects = scene?.objects || []
  const relationships = scene?.relationships || []
  return {
    totalObjects: objects.length,
    totalRelationships: relationships.length,
    visibleObjects: visibleNodeIds(objects, hiddenObjectIds).size,
    visibleRelationships: visibleEdgeIds(relationships, hiddenObjectIds, hiddenRelationshipIds)
      .size,
  }
}

/**
 * Selection cleanup after a visibility change.
 *
 * The inspector must never keep showing an element that is no longer drawn, so
 * a selection is cleared exactly when the element it points at became
 * invisible:
 *
 *   - selected object hidden                      -> object selection cleared
 *   - selected relationship hidden, or one of its
 *     endpoints hidden                            -> relationship cleared
 *   - an UNRELATED element hidden                 -> selection untouched
 */
export function resolveSelectionAfterVisibilityChange({
  scene,
  selectedObjectId,
  selectedRelationshipId,
  hiddenObjectIds,
  hiddenRelationshipIds,
}) {
  let nextObjectId = selectedObjectId
  let nextRelationshipId = selectedRelationshipId

  if (nextObjectId && hiddenObjectIds.has(nextObjectId)) {
    nextObjectId = null
  }

  if (nextRelationshipId) {
    const rel = (scene?.relationships || []).find((r) => r.id === nextRelationshipId)
    if (!rel || !isRelationshipGraphVisible(rel, hiddenObjectIds, hiddenRelationshipIds)) {
      nextRelationshipId = null
    }
  }

  return { selectedObjectId: nextObjectId, selectedRelationshipId: nextRelationshipId }
}
