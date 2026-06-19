import fs from "fs";
import path from "path";
import { defineConfig, devices } from "@playwright/test";
import { STORAGE_STATE } from "./e2e/storage-state";

// Load e2e credentials from the gitignored frontend/.env.e2e without pulling in
// a dotenv dependency. Lines are simple KEY=VALUE; existing env vars win so the
// values can still be overridden from the shell/CI.
function loadEnvFile(file: string) {
  const full = path.resolve(__dirname, file);
  if (!fs.existsSync(full)) return;
  for (const line of fs.readFileSync(full, "utf8").split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const eq = trimmed.indexOf("=");
    if (eq === -1) continue;
    const key = trimmed.slice(0, eq).trim();
    const value = trimmed.slice(eq + 1).trim();
    if (!(key in process.env)) process.env[key] = value;
  }
}

loadEnvFile(".env.e2e");

export default defineConfig({
  testDir: "./e2e",
  timeout: 30_000,
  retries: 0,
  use: {
    baseURL: "http://localhost:3000",
    trace: "on-first-retry",
  },
  projects: [
    // Signs in once and writes the authenticated storage state.
    { name: "setup", testMatch: /auth\.setup\.ts/ },
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"], storageState: STORAGE_STATE },
      dependencies: ["setup"],
      // The setup file is itself a spec; exclude it from the test project.
      testIgnore: /auth\.setup\.ts/,
    },
  ],
  // Start the dev server with real auth enforced (no DEV_MODE bypass), so the
  // login flow and JWT-forwarding path are exercised end to end.
  webServer: {
    command: "npm run dev",
    url: "http://localhost:3000",
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
    env: { NEXT_PUBLIC_DEV_MODE: "false" },
  },
});
