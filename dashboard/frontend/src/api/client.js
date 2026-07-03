/* API client for the Smart Grid RL backend. */

const API = import.meta.env.VITE_API_URL
  ? `${import.meta.env.VITE_API_URL}/api`
  : '/api'  // fallback for local dev via Vite proxy

async function jsonFetch(url, options = {}) {
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`${res.status} ${res.statusText}: ${text}`)
  }
  return res.json()
}

export const api = {
  health: () => jsonFetch(`${API}/health`),
  feeders: () => jsonFetch(`${API}/feeders`),
  policies: () => jsonFetch(`${API}/policies`),
  simulate: (body) =>
    jsonFetch(`${API}/simulate`, { method: 'POST', body: JSON.stringify(body) }),
  compare: (body) =>
    jsonFetch(`${API}/compare`, { method: 'POST', body: JSON.stringify(body) }),
  stressTest: (body) =>
    jsonFetch(`${API}/stress_test`, { method: 'POST', body: JSON.stringify(body) }),
}

/* Shared display constants used across the pages. */

export const POLICY_COLORS = {
  ppo: '#2563EB',                 // brand blue
  ppo_robust: '#9333EA',          // brand purple
  demand_proportional: '#059669', // brand green
  equal_split: '#F59E0B',         // amber
  priority_based: '#EF4444',      // red
  deficit_chasing: '#6B7280',     // gray
}

export const BAND_COLORS = {
  A: '#1E40AF',
  B: '#3B82F6',
  C: '#60A5FA',
  D: '#A855F7',
  E: '#C4B5FD',
}

export const SCENARIO_LABELS = {
  easy: 'Easy (90%)',
  baseline: 'Baseline (70%)',
  hard: 'Hard (50%)',
  extreme: 'Extreme (30%)',
}
