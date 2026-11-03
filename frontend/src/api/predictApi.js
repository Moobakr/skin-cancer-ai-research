const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api';

export async function getHealth() {
  const response = await fetch(`${API_BASE_URL}/health/`);
  if (!response.ok) {
    throw new Error('Backend health check failed');
  }
  return response.json();
}

export async function predictSkinCancer(formData) {
  const response = await fetch(`${API_BASE_URL}/predict/`, {
    method: 'POST',
    body: formData
  });

  const payload = await response.json().catch(() => ({}));

  if (!response.ok) {
    const message = payload?.image?.[0] || payload?.error || 'Prediction request failed';
    throw new Error(message);
  }

  return payload;
}
