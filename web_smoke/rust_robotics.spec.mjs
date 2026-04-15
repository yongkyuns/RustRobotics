import { expect, test } from "@playwright/test";

async function bootApp(page, baseURL) {
  const consoleErrors = [];
  const pageErrors = [];

  page.on("console", (message) => {
    if (message.type() === "error" || message.type() === "assert") {
      consoleErrors.push(message.text());
    }
  });
  page.on("pageerror", (error) => {
    pageErrors.push(error.message);
  });

  const response = await page.goto(baseURL, { waitUntil: "domcontentloaded" });
  expect(response?.ok()).toBeTruthy();
  await expect(page).toHaveTitle(/Rust Robotics/);
  await expect(page.locator("#the_canvas_id")).toBeVisible();

  await page.waitForFunction(() => !document.getElementById("center_text"), null, {
    timeout: 60_000,
  });
  await page.waitForFunction(() => {
    const canvas = document.getElementById("the_canvas_id");
    return canvas instanceof HTMLCanvasElement && canvas.width > 0 && canvas.height > 0;
  });

  await page.waitForFunction(() => {
    return (
      typeof window.rustRoboticsTestGetState === "function" &&
      typeof window.rustRoboticsTestSetMode === "function" &&
      typeof window.rustRoboticsTestSetPaused === "function" &&
      typeof window.rustRoboticsTestRestart === "function"
    );
  });

  return { consoleErrors, pageErrors };
}

test("docs server exposes required isolation headers", async ({ request, baseURL }) => {
  const response = await request.get(baseURL);

  expect(response.ok()).toBeTruthy();
  expect(response.headers()["cross-origin-opener-policy"]).toBe("same-origin");
  expect(response.headers()["cross-origin-embedder-policy"]).toBe("require-corp");
  expect(response.headers()["cross-origin-resource-policy"]).toBe("same-origin");
});

test("web app boots without page or console errors", async ({ page, baseURL }) => {
  const { consoleErrors, pageErrors } = await bootApp(page, baseURL);

  const bridgeStatus = await page.evaluate(() => ({
    hasWasmBindgen: typeof globalThis.wasm_bindgen === "object",
    hasPpoTrainerCreate: typeof globalThis.rustRoboticsPpoTrainerCreate === "function",
    hasMujocoInit: typeof window.rustRoboticsMujocoInit === "function",
    hasMujocoStep: typeof window.rustRoboticsMujocoStep === "function",
    hasTestGetState: typeof window.rustRoboticsTestGetState === "function",
    hasTestSetMode: typeof window.rustRoboticsTestSetMode === "function",
    hasTestSetPaused: typeof window.rustRoboticsTestSetPaused === "function",
    hasTestRestart: typeof window.rustRoboticsTestRestart === "function",
  }));

  expect(bridgeStatus).toEqual({
    hasWasmBindgen: true,
    hasPpoTrainerCreate: true,
    hasMujocoInit: true,
    hasMujocoStep: true,
    hasTestGetState: true,
    hasTestSetMode: true,
    hasTestSetPaused: true,
    hasTestRestart: true,
  });

  await page.waitForTimeout(1_000);
  expect(pageErrors).toEqual([]);
  expect(consoleErrors).toEqual([]);
});

test("browser test hooks can switch mode pause and restart deterministically", async ({
  page,
  baseURL,
}) => {
  const { consoleErrors, pageErrors } = await bootApp(page, baseURL);

  const readState = () => page.evaluate(() => window.rustRoboticsTestGetState());

  await page.waitForFunction(() => {
    const state = window.rustRoboticsTestGetState();
    return state.mode === "inverted_pendulum" && state.paused === false;
  });

  await page.evaluate(() => window.rustRoboticsTestSetPaused(true));
  await page.waitForFunction(() => {
    const state = window.rustRoboticsTestGetState();
    return state.paused === true;
  });

  await page.evaluate(() => window.rustRoboticsTestSetMode("path_planning"));
  await page.waitForFunction(() => {
    const state = window.rustRoboticsTestGetState();
    return state.mode === "path_planning" && state.time === 0;
  });

  await page.evaluate(() => window.rustRoboticsTestSetPaused(false));
  await page.waitForFunction(() => {
    const state = window.rustRoboticsTestGetState();
    return state.mode === "path_planning" && state.paused === false;
  });
  await page.waitForFunction(() => {
    const state = window.rustRoboticsTestGetState();
    return state.time >= 0.05;
  });

  await page.evaluate(() => window.rustRoboticsTestSetPaused(true));
  await page.waitForFunction(() => {
    const state = window.rustRoboticsTestGetState();
    return state.paused === true;
  });

  const timeBeforeRestart = await page.evaluate(() => {
    const state = window.rustRoboticsTestGetState();
    return state.time;
  });
  expect(timeBeforeRestart).toBeGreaterThan(0.01);

  await page.evaluate(() => window.rustRoboticsTestRestart());
  await page.waitForFunction(() => {
    const state = window.rustRoboticsTestGetState();
    return state.mode === "path_planning" && state.paused === true && state.time === 0;
  });

  await expect.poll(readState).toMatchObject({
    mode: "path_planning",
    paused: true,
    time: 0,
  });
  expect(pageErrors).toEqual([]);
  expect(consoleErrors).toEqual([]);
});
