// Bundled neural models shipped with the web app: fetched at runtime from
// /models/models.json (owned by the AI workstream — this hook only reads it).
// A missing manifest or a failed fetch is normal before any model is bundled;
// the UI degrades to "no neural models found" and Master falls back to search.

import { useCallback, useEffect, useState } from 'react';
import type { ModelOption } from '../components/types';

export interface LocalModelManifestEntry {
  id: string;
  name: string;
  url: string;
  description?: string;
  trainedGames?: number;
  notes?: string;
}

export interface LocalModelsState {
  models: ModelOption[];
  loading: boolean;
  /** Non-null when the manifest could not be loaded. Not fatal. */
  error: string | null;
  refresh: () => void;
}

function normalizeEntry(raw: unknown): ModelOption | null {
  if (!raw || typeof raw !== 'object') return null;
  const e = raw as Record<string, unknown>;
  if (typeof e.id !== 'string' || typeof e.name !== 'string' || typeof e.url !== 'string') {
    return null;
  }
  return {
    id: e.id,
    name: e.name,
    url: e.url,
    source: 'bundled',
    description: typeof e.description === 'string' ? e.description : undefined,
    trainedGames: typeof e.trainedGames === 'number' ? e.trainedGames : undefined,
    notes: typeof e.notes === 'string' ? e.notes : undefined,
  };
}

export function useLocalModels(): LocalModelsState {
  const [models, setModels] = useState<ModelOption[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    fetch('/models/models.json')
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json() as Promise<unknown>;
      })
      .then((data) => {
        if (cancelled) return;
        const list =
          data && typeof data === 'object' && Array.isArray((data as { models?: unknown }).models)
            ? (data as { models: unknown[] }).models
            : [];
        setModels(list.map(normalizeEntry).filter((m): m is ModelOption => m !== null));
        setLoading(false);
      })
      .catch(() => {
        if (cancelled) return;
        setModels([]);
        setError('No neural models found — Master falls back to search.');
        setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [nonce]);

  const refresh = useCallback(() => setNonce((n) => n + 1), []);

  return { models, loading, error, refresh };
}
