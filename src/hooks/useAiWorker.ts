import { useCallback, useEffect, useRef, useState } from 'react';
import type { AiMoveRequest, AiMoveResult, AnalysisResult } from '../ai/types';
import type {
  AiWorkerError,
  AiWorkerRequest,
  AiWorkerResponse,
} from '../workers/ai.worker';

export interface UseAiWorker {
  requestMove: (req: AiMoveRequest) => Promise<AiMoveResult>;
  analyze: (req: AiMoveRequest) => Promise<AnalysisResult>;
  /**
   * 'idle' | 'thinking' | 'fallback' | 'error'.
   * 'fallback' means the Worker could not be created and the AI runs on the
   * main thread instead (works, but long searches will block the UI).
   */
  status: string;
  /** Cancel all pending requests: terminates the worker and recreates it. */
  cancel: () => void;
}

interface Pending {
  resolve: (value: AiMoveResult | AnalysisResult) => void;
  reject: (err: Error) => void;
}

export function useAiWorker(): UseAiWorker {
  const workerRef = useRef<Worker | null>(null);
  const pendingRef = useRef(new Map<number, Pending>());
  const idRef = useRef(0);
  const fallbackRef = useRef(false);
  const [status, setStatus] = useState('idle');

  const createWorker = useCallback(() => {
    try {
      const worker = new Worker(new URL('../workers/ai.worker.ts', import.meta.url), {
        type: 'module',
      });
      worker.onmessage = (event: MessageEvent<AiWorkerResponse | AiWorkerError>) => {
        const data = event.data;
        const pending = pendingRef.current.get(data.id);
        if (!pending) return;
        pendingRef.current.delete(data.id);
        if (pendingRef.current.size === 0) setStatus('idle');
        if (data.type === 'result') pending.resolve(data.result);
        else pending.reject(new Error(data.message));
      };
      worker.onerror = () => {
        // Worker crashed or failed to load: reject pending requests and
        // switch to main-thread fallback for subsequent requests.
        fallbackRef.current = true;
        workerRef.current = null;
        for (const [, p] of pendingRef.current) {
          p.reject(new Error('AI worker failed; falling back to main thread'));
        }
        pendingRef.current.clear();
        setStatus('fallback');
      };
      workerRef.current = worker;
      fallbackRef.current = false;
    } catch {
      // Worker construction unsupported (e.g. old browser, CSP): main thread.
      fallbackRef.current = true;
      workerRef.current = null;
      setStatus('fallback');
    }
  }, []);

  useEffect(() => {
    createWorker();
    return () => {
      workerRef.current?.terminate();
      workerRef.current = null;
      pendingRef.current.clear();
    };
  }, [createWorker]);

  const send = useCallback(
    async <T extends AiMoveResult | AnalysisResult>(
      type: 'move' | 'analyze',
      req: AiMoveRequest,
    ): Promise<T> => {
      if (fallbackRef.current || !workerRef.current) {
        // Main-thread fallback: dynamic import so the AI bundle is only
        // loaded on the main thread when the worker is unavailable.
        setStatus('fallback');
        const { chooseMove, analyzePosition } = await import('../ai/engine');
        const result =
          type === 'analyze' ? await analyzePosition(req) : await chooseMove(req);
        return result as T;
      }
      const id = ++idRef.current;
      setStatus('thinking');
      const request: AiWorkerRequest = { type, id, payload: req };
      return new Promise<T>((resolve, reject) => {
        pendingRef.current.set(id, {
          resolve: resolve as (value: AiMoveResult | AnalysisResult) => void,
          reject,
        });
        workerRef.current?.postMessage(request);
      });
    },
    [],
  );

  const requestMove = useCallback(
    (req: AiMoveRequest) => send<AiMoveResult>('move', req),
    [send],
  );

  const analyze = useCallback(
    (req: AiMoveRequest) => send<AnalysisResult>('analyze', req),
    [send],
  );

  const cancel = useCallback(() => {
    for (const [, p] of pendingRef.current) {
      p.reject(new Error('AI request cancelled'));
    }
    pendingRef.current.clear();
    workerRef.current?.terminate();
    workerRef.current = null;
    // Recreate so the next request has a fresh worker (unless we're in
    // fallback mode, where there is no worker to recreate).
    if (!fallbackRef.current) createWorker();
    setStatus(fallbackRef.current ? 'fallback' : 'idle');
  }, [createWorker]);

  return { requestMove, analyze, status, cancel };
}
