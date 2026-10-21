import * as ort from 'onnxruntime-web'

// Public return value remains exactly { image, objects, relationships }.
const configuredBase = import.meta.env.VITE_BROWSER_MODEL_BASE_URL || 'browser-models/scene-graph-v1/'
const MODEL_BASE_URL = new URL(configuredBase, new URL(import.meta.env.BASE_URL, window.location.origin)).toString().replace(/\/+$/, '/')
const ASSETS = { yolo: 'model/yolov8s-world-vg150-640-static-adaptive-max.onnx', clip: 'model/d1-fp16-clip-vit-b-32-openai-visual-fp16-io-fp32.onnx', relationship: 'model/d1-relationship-mlp-full-fusion-fp32-dynamic.onnx', vocabulary: 'references/vocabulary.json', mappings: 'e1/references/mappings.json', wasm: 'ort/' }
const sessionPromises = new Map()
let metadataPromise
let activeAnalysis = false
let worker
let workerSequence = 0
const workerRequests = new Map()

function assetUrl(name) { return new URL(ASSETS[name], MODEL_BASE_URL).toString() }
function report(onProgress, stage, detail, provider) { onProgress?.({ stage, detail, provider }) }
function timing(stage, durationMs) { window.dispatchEvent(new CustomEvent('scene-graph-inference-timing', { detail: { stage, durationMs } })) }
function friendlyError(error) { const message = error instanceof Error ? error.message : String(error); if (/createImageBitmap|image|decode|bitmap/i.test(message)) return new Error('This image could not be decoded. Choose a valid JPEG, PNG, or WEBP image.'); if (/fetch|HTTP 404|HTTP 403|HTTP 5\d\d|model asset|Failed to fetch/i.test(message)) return new Error(`Could not load a required browser model asset. ${message}`); return new Error(`Browser analysis failed: ${message}`) }
function requireBrowserSupport() { if (typeof Worker === 'undefined' || typeof OffscreenCanvas === 'undefined' || typeof createImageBitmap === 'undefined') throw new Error('This browser does not support the local worker and image-processing APIs required for browser inference.') }
async function fetchRequired(url, description) { const response = await fetch(url, { cache: 'force-cache', mode: 'cors' }); if (!response.ok) throw new Error(`${description} fetch failed (HTTP ${response.status})`); return response }

async function loadMetadata() {
  if (!metadataPromise) metadataPromise = Promise.all([fetchRequired(assetUrl('vocabulary'), 'VG150 vocabulary').then((response) => response.json()), fetchRequired(assetUrl('mappings'), 'Visual Genome mappings').then((response) => response.json())]).then(([vocabulary, mappings]) => ({ labels: Array.isArray(vocabulary) ? vocabulary : vocabulary.vocabulary, labelIds: mappings.label_name_to_vg_id, predicates: mappings.model_class_to_predicate }))
  try { return await metadataPromise } catch (error) { metadataPromise = undefined; throw error }
}

async function loadSession(name, provider) {
  const cacheKey = `${name}:${provider}`
  if (!sessionPromises.has(cacheKey)) sessionPromises.set(cacheKey, (async () => { const response = await fetchRequired(assetUrl(name), `${name} model`); return ort.InferenceSession.create(await response.arrayBuffer(), { executionProviders: [provider] }) })())
  try { return await sessionPromises.get(cacheKey) } catch (error) { sessionPromises.delete(cacheKey); throw error }
}

async function loadClipSession(onProgress) {
  if (navigator.gpu) {
    try { report(onProgress, 'Loading AI models', 'Initializing OpenCLIP with WebGPU. First-time setup can take a while…', 'webgpu'); return { session: await loadSession('clip', 'webgpu'), provider: 'webgpu' } } catch (error) { console.warn('WebGPU OpenCLIP initialization failed; using WASM fallback.', error); report(onProgress, 'Loading AI models', 'WebGPU initialization failed; using the slower WASM fallback…', 'wasm') }
  } else report(onProgress, 'Loading AI models', 'WebGPU is unavailable; using the slower WASM fallback for OpenCLIP…', 'wasm')
  return { session: await loadSession('clip', 'wasm'), provider: 'wasm' }
}

function inferenceWorker() {
  if (!worker) {
    worker = new Worker(new URL('./inferenceWorker.js', import.meta.url), { type: 'module' })
    worker.onmessage = ({ data }) => { const request = workerRequests.get(data.id); if (!request) return; workerRequests.delete(data.id); if (data.error) request.reject(new Error(data.error)); else { timing(data.stage, data.durationMs); request.resolve(data) } }
    worker.onerror = (event) => { for (const { reject } of workerRequests.values()) reject(new Error(event.message || 'Browser preprocessing worker failed.')); workerRequests.clear() }
  }
  return worker
}
function runWorker(type, payload, transfer = []) { const id = ++workerSequence; return new Promise((resolve, reject) => { workerRequests.set(id, { resolve, reject }); inferenceWorker().postMessage({ id, type, ...payload }, transfer) }) }
function timedRun(stage, work) { const started = performance.now(); return work().then((result) => { timing(stage, performance.now() - started); return result }) }

export async function analyzeImage(file, onProgress) {
  if (activeAnalysis) throw new Error('An image analysis is already in progress. Wait for it to finish before starting another one.')
  if (!(file instanceof File)) throw new Error('Choose an image file before running analysis.')
  if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) throw new Error('Unsupported image format. Choose a JPEG, PNG, or WEBP image.')
  activeAnalysis = true
  try {
    requireBrowserSupport()
    report(onProgress, 'Loading AI models', 'Loading model metadata and reusable inference sessions. First-time setup may be slow…')
    ort.env.wasm.wasmPaths = assetUrl('wasm'); ort.env.wasm.numThreads = 1
    const sessionsStarted = performance.now()
    const [metadata, yolo, relationship, clip, preparedYolo] = await Promise.all([loadMetadata(), loadSession('yolo', 'wasm'), loadSession('relationship', 'wasm'), loadClipSession(onProgress), runWorker('prepare-yolo', { file })])
    timing('modelAndMetadataInitialization', performance.now() - sessionsStarted)
    report(onProgress, 'Detecting objects', 'Running YOLO-World locally in this browser…', 'wasm')
    const yoloOutput = await timedRun('yoloInference', () => yolo.run({ [yolo.inputNames[0]]: new ort.Tensor('float32', new Float32Array(preparedYolo.inputBuffer), [1, 3, 640, 640]) }))
    report(onProgress, 'Preparing object relationships', 'Running detection postprocessing and preparing unique union crops…')
    const preparedRelationships = await runWorker('prepare-relationships', { bitmap: preparedYolo.bitmap, rawBuffer: yoloOutput[yolo.outputNames[0]].data.buffer, input: preparedYolo.input, width: preparedYolo.width, height: preparedYolo.height, metadata }, [preparedYolo.bitmap, yoloOutput[yolo.outputNames[0]].data.buffer])
    if (!preparedRelationships.pairs.length) { report(onProgress, 'Building scene graph', 'No valid directed object pairs were produced.'); report(onProgress, 'Complete', 'Analysis complete.', clip.provider); return { image: { width: preparedYolo.width, height: preparedYolo.height }, objects: preparedRelationships.objects, relationships: [] } }
    const visualInput = new Float32Array(preparedRelationships.visualInputBuffer)
    report(onProgress, 'Extracting visual features', `Encoding ${preparedRelationships.groupRows.length} unique union crop${preparedRelationships.groupRows.length === 1 ? '' : 's'} with OpenCLIP…`, clip.provider)
    const uniqueOutput = await timedRun('clipInference', () => clip.session.run({ [clip.session.inputNames[0]]: new ort.Tensor('float32', visualInput, [preparedRelationships.groupRows.length, 3, 224, 224]) }))
    const uniqueFeatures = uniqueOutput[clip.session.outputNames[0]].data, visual = new Float32Array(preparedRelationships.pairs.length * 512)
    preparedRelationships.groupRows.forEach((rows, uniqueRow) => rows.forEach((row) => visual.set(uniqueFeatures.subarray(uniqueRow * 512, uniqueRow * 512 + 512), row * 512)))
    report(onProgress, 'Predicting relationships', `Running the relationship model for ${preparedRelationships.pairs.length} directed pair${preparedRelationships.pairs.length === 1 ? '' : 's'}…`, 'wasm')
    const subject = new BigInt64Array(preparedRelationships.pairs.map((pair) => BigInt(pair.subject.label_index - 1))), object = new BigInt64Array(preparedRelationships.pairs.map((pair) => BigInt(pair.object.label_index - 1)))
    const geo = new Float32Array(preparedRelationships.pairs.flatMap((pair) => { const unionX = pair.union.cx - pair.union.width / 2, unionY = pair.union.cy - pair.union.height / 2, s = pair.subject.bbox, o = pair.object.bbox; return [(s.cx - unionX) / pair.union.width, (s.cy - unionY) / pair.union.height, s.width / pair.union.width, s.height / pair.union.height, (o.cx - unionX) / pair.union.width, (o.cy - unionY) / pair.union.height, o.width / pair.union.width, o.height / pair.union.height] }))
    const relationshipOutput = await timedRun('relationshipInference', () => relationship.run({ visual: new ort.Tensor('float32', visual, [preparedRelationships.pairs.length, 512]), subject_label: new ort.Tensor('int64', subject, [preparedRelationships.pairs.length]), object_label: new ort.Tensor('int64', object, [preparedRelationships.pairs.length]), geometry: new ort.Tensor('float32', geo, [preparedRelationships.pairs.length, 8]) }))
    report(onProgress, 'Building scene graph', 'Filtering and deduplicating predicted relationships…')
    const result = await runWorker('build-scene', { width: preparedYolo.width, height: preparedYolo.height, objects: preparedRelationships.objects, pairs: preparedRelationships.pairs, predicates: metadata.predicates, logitsBuffer: relationshipOutput.logits.data.buffer }, [relationshipOutput.logits.data.buffer])
    report(onProgress, 'Complete', 'Analysis complete.', clip.provider)
    return result.scene
  } catch (error) { throw friendlyError(error) } finally { activeAnalysis = false }
}

export const browserModelAssetBaseUrl = MODEL_BASE_URL