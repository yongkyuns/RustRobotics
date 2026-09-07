import { expect, test } from "@playwright/test";

test("focused pendulum MPC advances after tuning, switching, and restart", async ({
  page,
  baseURL,
}) => {
  const consoleErrors = [];
  const pageErrors = [];
  page.on("console", (message) => {
    if (message.type() === "error" || message.type() === "assert") {
      consoleErrors.push(message.text());
    }
  });
  page.on("pageerror", (error) => pageErrors.push(error.message));

  const response = await page.goto(
    `${baseURL}/?mode=inverted_pendulum&embed=focused&ui=smoke_mpc`,
    { waitUntil: "domcontentloaded" },
  );
  expect(response?.ok()).toBeTruthy();
  await page.waitForFunction(() => !document.getElementById("center_text"), null, {
    timeout: 60_000,
  });
  await page.waitForFunction(() => {
    return (
      typeof window.rustRoboticsEmbedGetState === "function" &&
      typeof window.rustRoboticsTestSetPaused === "function"
    );
  });
  await page.evaluate(() => window.rustRoboticsTestSetPaused(true));
  await page.waitForFunction(() => {
    const state = window.rustRoboticsEmbedGetState();
    return state?.mode === "inverted_pendulum" && state.paused === true &&
      state.payload?.kind === "pendulum";
  });

  const readState = () => page.evaluate(() => window.rustRoboticsEmbedGetState());
  const card = page.locator(".pendulum-dom-card").first();
  const controllerSelect = card.locator("select");
  await expect(card).toBeVisible();
  await page.locator("#toolbar_noise_enabled").uncheck();
  await expect.poll(async () => (await readState()).payload.noise_enabled).toBe(false);

  async function selectController(kind) {
    await controllerSelect.selectOption(kind);
    await expect.poll(async () => {
      return (await readState()).payload.pendulums[0].controller;
    }).toBe(kind);
  }

  async function restartPaused() {
    await page.locator("#toolbar_restart_button").click();
    await expect.poll(readState).toMatchObject({ paused: true, time: 0 });
  }

  // Native unit tests count cache constructions. Here exercise the real WASM
  // control path and DOM actions, rather than merely checking that the app boots.
  async function runAndPause(kind) {
    const before = await readState();
    expect(before.paused).toBe(true);
    await page.locator("#toolbar_pause_button").click();
    await page.waitForFunction((startTime) => {
      const state = window.rustRoboticsEmbedGetState();
      return state.paused === false && state.time >= startTime + 0.12;
    }, before.time);
    await page.locator("#toolbar_pause_button").click();
    await expect.poll(async () => (await readState()).paused).toBe(true);

    const after = await readState();
    expect(Number.isFinite(after.time)).toBe(true);
    expect(after.time).toBeGreaterThan(before.time);
    const pendulum = after.payload.pendulums[0];
    expect(pendulum.controller).toBe(kind);
    expect(pendulum).toHaveProperty("control_error");
    expect(pendulum.control_error ?? null).toBeNull();
    expect(pageErrors).toEqual([]);
    expect(consoleErrors).toEqual([]);
  }

  await selectController("mpc");
  await restartPaused();
  await runAndPause("mpc");

  await card.locator(".controller-params-section summary").click();
  const angleWeight = card.locator('[data-field-key="mpc_q_angle"] input');
  await angleWeight.fill("15");
  await angleWeight.blur();
  await expect.poll(async () => {
    return (await readState()).payload.pendulums[0].controller_params.q_angle;
  }).toBe(15);
  await runAndPause("mpc");

  await selectController("lqr");
  await runAndPause("lqr");
  await selectController("mpc");
  await runAndPause("mpc");

  await restartPaused();
  await runAndPause("mpc");
});
