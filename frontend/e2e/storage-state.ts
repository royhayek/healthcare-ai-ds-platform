/**
 * Shared path to the saved authenticated browser state.
 *
 * Kept in its own module (not the setup spec) so playwright.config.ts can
 * import it without loading a file that calls `test()`/`setup()` at import time.
 */

import path from "path"

export const STORAGE_STATE = path.resolve(__dirname, ".auth/user.json")
