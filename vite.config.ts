import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// onnxruntime-web ships WASM binaries; vite-plugin-static-copy is not
// required for the basic setup — ort's default CDN-free bundling resolves
// the wasm files from node_modules via import.meta.url in recent versions.
// If WASM loading fails in a deployment, copy node_modules/onnxruntime-web/dist/*.wasm
// into public/ and set ort.env.wasm.wasmPaths (see src/ai/onnxModel.ts).
export default defineConfig({
  plugins: [react()],
  worker: {
    format: 'es',
  },
  build: {
    target: 'es2022',
  },
});
