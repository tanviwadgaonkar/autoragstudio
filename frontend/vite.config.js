import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// /studio/* -> Django (8000), /engine/* -> FastAPI (8001).
// Keeps the browser same-origin, so no CORS setup is needed in dev.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/studio": {
        target: "http://localhost:8000",
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/studio/, ""),
      },
      "/engine": {
        target: "http://localhost:8001",
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/engine/, ""),
      },
    },
  },
});
