import { defineConfig } from "vitest/config"
import react from "@vitejs/plugin-react"
import path from "path"

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      // Mirror the tsconfig "@/*" -> "./*" path mapping.
      "@": path.resolve(__dirname, "."),
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./vitest.setup.ts"],
    // Playwright specs live in e2e/ and are run by `npm run e2e`, not vitest.
    include: ["**/*.test.{ts,tsx}"],
    exclude: ["node_modules", ".next", "e2e", "dist"],
  },
})
