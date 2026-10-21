const CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
const CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

function bankersRound(value) { const floor = Math.floor(value); const fraction = value - floor; if (fraction < 0.5) return floor; if (fraction > 0.5) return floor + 1; return floor % 2 === 0 ? floor : floor + 1 }
function boxFromXyxy(x1, y1, x2, y2) { return { cx: (x1 + x2) / 2, cy: (y1 + y2) / 2, width: x2 - x1, height: y2 - y1 } }
function xyxy(box) { return [box.cx - box.width / 2, box.cy - box.height / 2, box.cx + box.width / 2, box.cy + box.height / 2] }
function iou(a, b) { const x1 = Math.max(a[0], b[0]); const y1 = Math.max(a[1], b[1]); const x2 = Math.min(a[2], b[2]); const y2 = Math.min(a[3], b[3]); const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1); const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]); const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]); return intersection / Math.max(areaA + areaB - intersection, 1e-30) }

function yoloInput(bitmap) {
  const ratio = Math.min(640 / bitmap.width, 640 / bitmap.height), resizedWidth = bankersRound(bitmap.width * ratio), resizedHeight = bankersRound(bitmap.height * ratio), left = bankersRound((640 - resizedWidth) / 2 - 0.1), top = bankersRound((640 - resizedHeight) / 2 - 0.1)
  const canvas = new OffscreenCanvas(640, 640), context = canvas.getContext('2d', { willReadFrequently: true })
  context.fillStyle = 'rgb(114,114,114)'; context.fillRect(0, 0, 640, 640); context.imageSmoothingEnabled = true; context.imageSmoothingQuality = 'high'; context.drawImage(bitmap, 0, 0, bitmap.width, bitmap.height, left, top, resizedWidth, resizedHeight)
  const pixels = context.getImageData(0, 0, 640, 640).data, data = new Float32Array(3 * 640 * 640)
  for (let index = 0; index < 640 * 640; index += 1) { data[index] = pixels[index * 4] / 255; data[640 * 640 + index] = pixels[index * 4 + 1] / 255; data[2 * 640 * 640 + index] = pixels[index * 4 + 2] / 255 }
  return { data, ratio, left, top }
}

function decodeYolo(raw, input, width, height, metadata) {
  const candidates = [], proposalCount = 8400
  for (let index = 0; index < proposalCount; index += 1) {
    let classId = 0, confidence = raw[4 * proposalCount + index]
    for (let classIndex = 1; classIndex < 149; classIndex += 1) { const score = raw[(4 + classIndex) * proposalCount + index]; if (score > confidence) { confidence = score; classId = classIndex } }
    if (confidence < 0.25) continue
    const cx = raw[index], cy = raw[proposalCount + index], boxWidth = raw[2 * proposalCount + index], boxHeight = raw[3 * proposalCount + index]
    candidates.push({ classId, confidence, box: [Math.max(0, Math.min(width, (cx - boxWidth / 2 - input.left) / input.ratio)), Math.max(0, Math.min(height, (cy - boxHeight / 2 - input.top) / input.ratio)), Math.max(0, Math.min(width, (cx + boxWidth / 2 - input.left) / input.ratio)), Math.max(0, Math.min(height, (cy + boxHeight / 2 - input.top) / input.ratio))] })
  }
  candidates.sort((a, b) => b.confidence - a.confidence)
  const kept = []
  for (const candidate of candidates) { if (kept.filter((item) => item.classId === candidate.classId).some((item) => iou(item.box, candidate.box) > 0.7)) continue; kept.push(candidate); if (kept.length >= 300) break }
  return kept.map((item, index) => { const label = metadata.labels[item.classId], labelIndex = metadata.labelIds[label]; if (!label || !labelIndex) throw new Error(`VG mapping unavailable for YOLO class ${item.classId}`); return { id: `obj_${index}`, label, label_index: labelIndex, confidence: item.confidence, bbox: boxFromXyxy(...item.box) } })
}

function pairRows(objects, imageWidth, imageHeight) {
  const pairs = []
  for (let subjectIndex = 0; subjectIndex < objects.length; subjectIndex += 1) for (let objectIndex = 0; objectIndex < objects.length; objectIndex += 1) {
    const subject = objects[subjectIndex], object = objects[objectIndex]
    if (subjectIndex === objectIndex || subject.label === object.label) continue
    const expandedSubject = { ...subject.bbox, width: subject.bbox.width * 1.2, height: subject.bbox.height * 1.2 }, expandedObject = { ...object.bbox, width: object.bbox.width * 1.2, height: object.bbox.height * 1.2 }
    const [sx1, sy1, sx2, sy2] = xyxy(expandedSubject), [ox1, oy1, ox2, oy2] = xyxy(expandedObject), x1 = Math.min(sx1, ox1), y1 = Math.min(sy1, oy1), x2 = Math.max(sx2, ox2), y2 = Math.max(sy2, oy2)
    const cropBounds = [Math.max(0, Math.min(imageWidth, bankersRound(x1))), Math.max(0, Math.min(imageHeight, bankersRound(y1))), Math.max(0, Math.min(imageWidth, bankersRound(x2))), Math.max(0, Math.min(imageHeight, bankersRound(y2)))]
    if (cropBounds[2] <= cropBounds[0] || cropBounds[3] <= cropBounds[1]) throw new Error(`Invalid relationship crop ${cropBounds}`)
    pairs.push({ subject, object, union: boxFromXyxy(x1, y1, x2, y2), cropBounds })
  }
  return pairs
}

function clipTensor(bitmap, bounds) {
  const cropWidth = bounds[2] - bounds[0], cropHeight = bounds[3] - bounds[1], scale = 224 / Math.min(cropWidth, cropHeight), resizedWidth = Math.max(224, bankersRound(cropWidth * scale)), resizedHeight = Math.max(224, bankersRound(cropHeight * scale))
  const resized = new OffscreenCanvas(resizedWidth, resizedHeight), resizedContext = resized.getContext('2d', { willReadFrequently: true }); resizedContext.imageSmoothingEnabled = true; resizedContext.imageSmoothingQuality = 'high'; resizedContext.drawImage(bitmap, bounds[0], bounds[1], cropWidth, cropHeight, 0, 0, resizedWidth, resizedHeight)
  const output = new OffscreenCanvas(224, 224), outputContext = output.getContext('2d', { willReadFrequently: true }); outputContext.drawImage(resized, bankersRound((resizedWidth - 224) / 2), bankersRound((resizedHeight - 224) / 2), 224, 224, 0, 0, 224, 224)
  const pixels = outputContext.getImageData(0, 0, 224, 224).data, tensor = new Float32Array(3 * 224 * 224)
  for (let index = 0; index < 224 * 224; index += 1) for (let channel = 0; channel < 3; channel += 1) tensor[channel * 224 * 224 + index] = (pixels[index * 4 + channel] / 255 - CLIP_MEAN[channel]) / CLIP_STD[channel]
  return tensor
}

function stableSoftmax(logits) { const output = new Float32Array(logits.length); for (let row = 0; row < logits.length / 51; row += 1) { const offset = row * 51; let maximum = -Infinity; for (let column = 0; column < 51; column += 1) maximum = Math.max(maximum, logits[offset + column]); let sum = 0; for (let column = 0; column < 51; column += 1) sum += Math.exp(logits[offset + column] - maximum); for (let column = 0; column < 51; column += 1) output[offset + column] = Math.exp(logits[offset + column] - maximum) / sum } return output }
function buildScene(width, height, objects, pairs, probabilities, predicates) { const seen = new Set(), relationships = []; for (let row = 0; row < pairs.length; row += 1) { let classId = 0; for (let classIndex = 1; classIndex < 51; classIndex += 1) if (probabilities[row * 51 + classIndex] > probabilities[row * 51 + classId]) classId = classIndex; const confidence = probabilities[row * 51 + classId]; if (classId === 50 || confidence < 0.3) continue; const pair = pairs[row], predicate = predicates[String(classId)], key = `${pair.subject.id}|${predicate}|${pair.object.id}`; if (seen.has(key)) continue; seen.add(key); relationships.push({ id: `rel_${relationships.length}`, subject_id: pair.subject.id, predicate, object_id: pair.object.id, confidence }) } return { image: { width, height }, objects: objects.map(({ id, label, confidence, bbox }) => ({ id, label, confidence, bbox })), relationships } }
function respond(id, stage, started, payload, transfer = []) { self.postMessage({ id, stage, durationMs: performance.now() - started, ...payload }, transfer) }

self.onmessage = async ({ data }) => {
  const started = performance.now()
  try {
    if (data.type === 'prepare-yolo') { const bitmap = await createImageBitmap(data.file, { imageOrientation: 'from-image' }), input = yoloInput(bitmap); respond(data.id, 'imageDecodeAndYoloPreprocess', started, { width: bitmap.width, height: bitmap.height, bitmap, input: { ratio: input.ratio, left: input.left, top: input.top }, inputBuffer: input.data.buffer }, [bitmap, input.data.buffer]); return }
    if (data.type === 'prepare-relationships') { const objects = decodeYolo(new Float32Array(data.rawBuffer), data.input, data.width, data.height, data.metadata), pairs = pairRows(objects, data.width, data.height); if (!pairs.length) { data.bitmap.close(); respond(data.id, 'yoloPostprocessAndRelationshipPreparation', started, { objects, pairs, groupRows: [], visualInputBuffer: null }); return } const groups = new Map(); pairs.forEach((pair, row) => { const key = pair.cropBounds.join(','), group = groups.get(key); if (group) group.rows.push(row); else groups.set(key, { bounds: pair.cropBounds, rows: [row] }) }); const visualInput = new Float32Array(groups.size * 3 * 224 * 224), groupRows = []; [...groups.values()].forEach((group, row) => { visualInput.set(clipTensor(data.bitmap, group.bounds), row * 3 * 224 * 224); groupRows.push(group.rows) }); data.bitmap.close(); respond(data.id, 'yoloPostprocessAndRelationshipPreparation', started, { objects, pairs, groupRows, visualInputBuffer: visualInput.buffer }, [visualInput.buffer]); return }
    if (data.type === 'build-scene') { respond(data.id, 'relationshipPostprocessAndSceneBuild', started, { scene: buildScene(data.width, data.height, data.objects, data.pairs, stableSoftmax(new Float32Array(data.logitsBuffer)), data.predicates) }); return }
    throw new Error(`Unknown inference worker action: ${data.type}`)
  } catch (error) { self.postMessage({ id: data.id, error: error instanceof Error ? error.message : String(error) }) }
}