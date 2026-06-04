/**
 * Step 4.5 happy-path: navigate to /, create a project, upload telco_churn.csv,
 * see the dataset listed.
 *
 * Requires a running backend (docker-compose up) and the Next.js dev server.
 * The playwright.config.ts webServer block starts Next.js automatically.
 */

import path from "path"
import { test, expect } from "@playwright/test"

const FIXTURE = path.resolve(__dirname, "../../backend/tests/fixtures/telco_churn.csv")

test("create project → upload dataset → see it listed", async ({ page }) => {
  // 1. Land on the projects list
  await page.goto("/")
  await expect(page.getByRole("heading", { name: /projects/i })).toBeVisible()

  // 2. Open "New project" dialog
  await page.getByTestId("new-project-button").click()
  await expect(page.getByRole("dialog")).toBeVisible()

  // 3. Fill and submit
  await page.getByTestId("project-name-input").fill("Telco Churn Test")
  await page.getByTestId("create-project-submit").click()

  // 4. Dialog closes and we're navigated to the project page
  await expect(page).toHaveURL(/\/project\/[a-z0-9-]+$/, { timeout: 10_000 })

  // 5. Upload dataset via the hidden file input
  const fileInput = page.locator('[data-testid="dataset-file-input"]')
  await fileInput.setInputFiles(FIXTURE)

  // 6. Dataset row appears in the table
  await expect(page.getByText("telco_churn.csv")).toBeVisible({ timeout: 15_000 })
})
