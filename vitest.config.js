import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    setupFiles: ["vitest.setup.js", "dotenv/config"],
    // Integration test gula (real API call kore) default e off.
    // `yarn test:int` diye chalao.
    exclude: ["**/node_modules/**", "**/*.int.test.js"],
  },
});
