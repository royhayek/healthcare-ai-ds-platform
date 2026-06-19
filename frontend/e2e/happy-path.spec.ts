/**
 * Happy path: land on /, create a project, open it, upload telco_churn.csv,
 * and see the dataset listed.
 *
 * Requires:
 *   - Postgres + Redis      → `docker compose up -d postgres redis`
 *   - Backend (FastAPI)     → uvicorn on :8001 with DEV_MODE=false and
 *                             SUPABASE_JWT_SECRET set, so the Bearer token this
 *                             test sends is actually verified.
 *   - Real auth             → the `setup` project signs in via the Supabase
 *                             login UI (E2E_EMAIL / E2E_PASSWORD in .env.e2e)
 *                             and saves the session; this test reuses it.
 * The playwright.config.ts webServer block starts the Next.js dev server with
 * NEXT_PUBLIC_DEV_MODE=false so the auth gate is enforced.
 *
 * The project name is unique per run so repeated runs against the persistent
 * dev database don't collide (the projects list shows every prior project).
 */

import path from "path"
import { test, expect } from "@playwright/test"

const FIXTURE = path.resolve(__dirname, "../../backend/tests/fixtures/telco_churn.csv")

test("create project → open it → upload dataset → see it listed", async ({ page }) => {
  const projectName = `Telco Churn E2E ${Date.now()}`

  // 1. Land on the projects list.
  await page.goto("/")
  await expect(page.getByRole("heading", { name: /projects/i })).toBeVisible()

  // 2. Open the "New project" dialog.
  await page.getByTestId("new-project-button").click()
  await expect(page.getByRole("dialog")).toBeVisible()

  // 3. Fill the name and submit.
  await page.getByTestId("project-name-input").fill(projectName)
  await page.getByTestId("create-project-submit").click()

  // 4. The dialog closes and the new project appears in the list.
  await expect(page.getByRole("dialog")).toBeHidden()
  const projectLink = page.getByRole("link", { name: new RegExp(projectName) })
  await expect(projectLink).toBeVisible({ timeout: 10_000 })

  // 5. Open the project.
  await projectLink.click()
  await expect(page).toHaveURL(/\/project\/[a-z0-9-]+$/, { timeout: 10_000 })

  // 6. Upload the dataset via the hidden file input.
  await page.locator('[data-testid="dataset-file-input"]').setInputFiles(FIXTURE)

  // 7. The dataset row appears.
  await expect(page.getByText("telco_churn.csv")).toBeVisible({ timeout: 15_000 })
})
