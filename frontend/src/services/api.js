// Historical import boundary retained so App and any future callers keep the
// same API. Inference now happens entirely in the current browser; no FastAPI
// request is made from the production frontend.
export { analyzeImage } from './browserInference.js'
