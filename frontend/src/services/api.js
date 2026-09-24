// API base URL is environment-driven (Deployment Phase 2).
// VITE_* variables are read by Vite at build/dev-server start time;
// the fallback preserves the exact local development behavior when no
// .env file or environment variable is present.  Production sets
// VITE_API_URL to the deployed backend origin (never hardcoded here).
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

export async function analyzeImage(file) {
  const formData = new FormData()
  formData.append('file', file)

  let response
  try {
    response = await fetch(`${API_BASE_URL}/analyze`, {
      method: 'POST',
      body: formData,
    })
  } catch (err) {
    throw new Error(
      `Could not reach the backend at ${API_BASE_URL}. Is it running? (${err.message})`,
    )
  }

  if (!response.ok) {
    let detail = response.statusText
    try {
      const errorBody = await response.json()
      if (errorBody && errorBody.detail) {
        detail = errorBody.detail
      }
    } catch {
      // ignore JSON parse errors
    }
    throw new Error(`Analysis failed (${response.status}): ${detail}`)
  }

  return response.json()
}
