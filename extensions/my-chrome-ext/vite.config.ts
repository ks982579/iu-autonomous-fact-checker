import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "path";

// CAUTION: very fragile - difficult to get working initially
// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "dist",
    minify: false,
    rollupOptions: {
      input: {
        popup: resolve(__dirname, "popup.html"),
      },
      output: {
        entryFileNames: "[name].js",
        chunkFileNames: "[name].js",
        assetFileNames: "[name].[ext]",
      },
    },
  },
  define: {
    'process.env.NODE_ENV': '"development"',
    '__DEV__': true,
  },
});

/**
 * from the input we removed:
 * - content: resolve(__dirname, "src/content/content.ts")
 * - background: resolve(__dirname, "src/background/background.ts")
 * - // TODO: Include below IF using React in content script; Else...
 * - contentPage: resolve(__dirname, "content.html")
 */
