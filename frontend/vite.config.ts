import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Two settings carry all the weight here.
//
// build.outDir  — the finished files land in ../static/app, which FastAPI already
//                 serves. Vite runs, writes, and exits; nothing of Vite is alive in
//                 production. The old page stays at / until this one replaces it.
//
// server.proxy  — during development the React dev server runs on :5173 and FastAPI
//                 on :8000. Two different ports are two different origins, and a
//                 browser blocks that by default. The proxy makes /api/... look local.
export default defineConfig({
  base: '/app/',
  plugins: [react()],
  build: {
    outDir: '../static/app',
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: { '/api': 'http://127.0.0.1:8000' },
  },
})
