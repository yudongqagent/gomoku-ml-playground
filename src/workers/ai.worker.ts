/// <reference lib="webworker" />
/**
 * AI Web Worker — thin wrapper around src/ai/engine.ts.
 *
 * Runs the (potentially seconds-long) search off the main thread so the UI
 * stays responsive. The ONNX model also loads inside this worker, which is
 * the recommended setup for onnxruntime-web.
 *
 * Message protocol:
 *   in:  { type: 'move' | 'analyze'; id: number; payload: AiMoveRequest }
 *   out: { type: 'result'; id: number; result: AiMoveResult | AnalysisResult }
 *        { type: 'error'; id: number; message: string }
 */

import { analyzePosition, chooseMove } from '../ai/engine';
import type { AiMoveRequest, AiMoveResult, AnalysisResult } from '../ai/types';

export interface AiWorkerRequest {
  type: 'move' | 'analyze';
  id: number;
  payload: AiMoveRequest;
}

export interface AiWorkerResponse {
  type: 'result';
  id: number;
  result: AiMoveResult | AnalysisResult;
}

export interface AiWorkerError {
  type: 'error';
  id: number;
  message: string;
}

const scope = self as unknown as DedicatedWorkerGlobalScope;

scope.onmessage = async (event: MessageEvent<AiWorkerRequest>) => {
  const { type, id, payload } = event.data;
  try {
    const result =
      type === 'analyze' ? await analyzePosition(payload) : await chooseMove(payload);
    const response: AiWorkerResponse = { type: 'result', id, result };
    scope.postMessage(response);
  } catch (err) {
    const response: AiWorkerError = {
      type: 'error',
      id,
      message: err instanceof Error ? err.message : String(err),
    };
    scope.postMessage(response);
  }
};
