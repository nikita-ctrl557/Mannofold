import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `base` is "/manifold/" for production builds so the bundle can be hosted
// under a subpath (e.g. statically inside the Studio site at /manifold/).
// Dev keeps "/" so the local Vite server + API proxy work unchanged.
export default defineConfig(({ command }) => ({
  base: command === "build" ? "/manifold/" : "/",
  plugins: [react()],
  server: {
    host: true,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/ws": {
        target: "ws://localhost:8000",
        ws: true,
        changeOrigin: true,
      },
    },
  },
}));
