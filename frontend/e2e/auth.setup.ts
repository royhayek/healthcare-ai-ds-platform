/**
 * Playwright auth setup — runs once before the test projects.
 *
 * Signs in through the real Supabase login UI (no DEV_MODE bypass) using the
 * credentials in E2E_EMAIL / E2E_PASSWORD, then saves the authenticated
 * browser state (cookies + localStorage) to disk. The chromium test project
 * loads that state via `storageState`, so every test runs as the real user and
 * its API calls carry a verifiable Supabase JWT (see lib/supabase.ts cookie
 * mirror and the proxy route).
 *
 * Credentials come from frontend/.env.e2e (gitignored), loaded in
 * playwright.config.ts. The setup fails loudly if they are missing rather than
 * silently falling back to an unauthenticated session.
 */

import { test as setup, expect } from "@playwright/test"
import { STORAGE_STATE } from "./storage-state"

setup("authenticate", async ({ page }) => {
  const email = process.env.E2E_EMAIL
  const password = process.env.E2E_PASSWORD

  expect(
    email && password,
    "E2E_EMAIL and E2E_PASSWORD must be set (see frontend/.env.e2e)",
  ).toBeTruthy()

  // Go straight to the login page and sign in via the real form.
  await page.goto("/login")
  await expect(page.getByRole("heading", { name: /sign in/i })).toBeVisible()

  await page.getByLabel("Email").fill(email!)
  await page.getByLabel("Password").fill(password!)
  await page.getByRole("button", { name: /sign in/i }).click()

  // A successful sign-in redirects to the projects list.
  await expect(page).toHaveURL(/\/$/, { timeout: 15_000 })
  await expect(page.getByRole("heading", { name: /projects/i })).toBeVisible({
    timeout: 15_000,
  })

  // Persist cookies + localStorage (includes the Supabase session and the
  // mirrored sb-access-token cookie the proxy forwards).
  await page.context().storageState({ path: STORAGE_STATE })
})
