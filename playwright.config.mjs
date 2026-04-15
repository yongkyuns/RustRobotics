import { defineConfig } from "@playwright/test";

const chromiumArgs = [
  "--enable-unsafe-webgpu",
  "--ignore-gpu-blocklist",
];

if (process.platform === "linux") {
  chromiumArgs.push("--disable-vulkan-surface");
}

export default defineConfig({
  testDir: "./web_smoke",
  timeout: 90_000,
  expect: {
    timeout: 60_000,
  },
  retries: process.env.CI ? 1 : 0,
  reporter: "list",
  use: {
    baseURL: "http://127.0.0.1:3000",
    browserName: "chromium",
    headless: true,
    launchOptions: {
      args: chromiumArgs,
    },
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    video: "retain-on-failure",
  },
  webServer: {
    command: "python3 scripts/serve_docs.py --host 127.0.0.1 --port 3000 --root docs",
    url: "http://127.0.0.1:3000/",
    reuseExistingServer: !process.env.CI,
    timeout: 30_000,
  },
});
