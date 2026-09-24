const API_BASE_URL = 'http://127.0.0.1:8000'

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
