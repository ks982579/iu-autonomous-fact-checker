import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "path";

// CAUTION: very fragile
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "dist",
    emptyOutDir: false,
    rollupOptions: {
      input: {
        content: resolve(__dirname, "src/content/content.tsx"),
      },
      output: {
        entryFileNames: "[name].js",
        chunkFileNames: "[name].js", 
        assetFileNames: "[name].[ext]",
        format: 'iife',
        name: 'ContentScript',
        inlineDynamicImports: true,
      },
    },
  },
  define: {
    'process.env.NODE_ENV': '"production"',
    '__DEV__': false,
    'process.env': '{}',
  }
});