import { expect, test } from "@playwright/test";

async function bootApp(page, url) {
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

  const response = await page.goto(url, { waitUntil: "domcontentloaded" });
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

test("focused path-planning embed boots with DOM controls and responds to planner actions", async ({
  page,
  baseURL,
}) => {
  const url = `${baseURL}/?mode=path_planning&embed=focused&ui=smoke_path_planning`;
  const { consoleErrors, pageErrors } = await bootApp(page, url);

  await page.waitForFunction(() => {
    const state = window.rustRoboticsEmbedGetState?.();
    return state?.mode === "path_planning" && state?.payload?.kind === "path_planning";
  });

  await expect(page.locator("#embed_path_toolbar")).toBeVisible();
  await expect(page.locator("#embed_path_planner_cards .planner-dom-card")).toHaveCount(1);
  await expect(page.locator("#path_toolbar_env_mode")).toHaveValue("grid");

  await page.selectOption(".planner-algorithm-select", "theta_star");
  await expect.poll(async () => {
    const state = await page.evaluate(() => window.rustRoboticsEmbedGetState());
    return state.payload.planners[0].algorithm;
  }).toBe("theta_star");

  await page.selectOption("#path_toolbar_env_mode", "continuous");
  await expect.poll(async () => {
    const state = await page.evaluate(() => window.rustRoboticsEmbedGetState());
    return state.payload.env_mode;
  }).toBe("continuous");

  await expect(page.locator("#path_toolbar_radius_wrap")).toBeVisible();
  await page.locator("#path_toolbar_add_planner_button").click();
  await expect(page.locator("#embed_path_planner_cards .planner-dom-card")).toHaveCount(2);

  const plannerCount = await page.locator("#path_toolbar_planner_count").textContent();
  expect(plannerCount).toContain("2 planner");
  expect(pageErrors).toEqual([]);
  expect(consoleErrors).toEqual([]);
});

test("focused localization embed boots with DOM controls and responds to vehicle actions", async ({
  page,
  baseURL,
}) => {
  const url = `${baseURL}/?mode=localization&embed=focused&ui=smoke_localization`;
  const { consoleErrors, pageErrors } = await bootApp(page, url);

  await page.waitForFunction(() => {
    const state = window.rustRoboticsEmbedGetState?.();
    return state?.mode === "localization" && state?.payload?.kind === "localization";
  });

  await expect(page.locator("#embed_localization_toolbar")).toBeVisible();
  await expect(page.locator("#embed_localization_cards .localization-dom-card")).toHaveCount(1);
  await expect(page.locator(".localization-drive-mode")).toHaveValue("kinematic");

  await page.locator(".localization-motion-details summary").click();
  await page.locator(".localization-velocity").fill("7.5");
  await page.locator(".localization-velocity").blur();
  await expect.poll(async () => {
    const state = await page.evaluate(() => window.rustRoboticsEmbedGetState());
    return Number(state.payload.vehicles[0].velocity).toFixed(1);
  }).toBe("7.5");

  await page.selectOption(".localization-drive-mode", "dynamic");
  await expect.poll(async () => {
    const state = await page.evaluate(() => window.rustRoboticsEmbedGetState());
    return state.payload.vehicles[0].drive_mode;
  }).toBe("dynamic");

  await page.locator("#localization_toolbar_add_vehicle_button").click();
  await expect(page.locator("#embed_localization_cards .localization-dom-card")).toHaveCount(2);

  const vehicleCount = await page.locator("#localization_toolbar_vehicle_count").textContent();
  expect(vehicleCount).toContain("2 vehicle");
  expect(pageErrors).toEqual([]);
  expect(consoleErrors).toEqual([]);
});

test("focused slam embed boots with DOM controls and responds to demo actions", async ({
  page,
  baseURL,
}) => {
  const url = `${baseURL}/?mode=slam&embed=focused&ui=smoke_slam`;
  const { consoleErrors, pageErrors } = await bootApp(page, url);

  await page.waitForFunction(() => {
    const state = window.rustRoboticsEmbedGetState?.();
    return state?.mode === "slam" && state?.payload?.kind === "slam";
  });

  await expect(page.locator("#embed_slam_toolbar")).toBeVisible();
  await expect(page.locator("#embed_slam_cards .slam-dom-card")).toHaveCount(1);
  await expect(page.locator(".slam-drive-mode")).toHaveValue("auto");

  await page.locator(".slam-ekf").uncheck();
  await expect.poll(async () => {
    const state = await page.evaluate(() => window.rustRoboticsEmbedGetState());
    return state.payload.demos[0].ekf_enabled;
  }).toBe(false);

  await page.selectOption(".slam-drive-mode", "manual");
  await expect.poll(async () => {
    const state = await page.evaluate(() => window.rustRoboticsEmbedGetState());
    return state.payload.demos[0].drive_mode;
  }).toBe("manual");

  await page
    .locator("#embed_slam_cards .slam-dom-card")
    .first()
    .locator("details")
    .nth(2)
    .locator("summary")
    .click();
  await page.locator(".slam-landmarks").fill("12");
  await page.locator(".slam-landmarks").blur();
  await expect.poll(async () => {
    const state = await page.evaluate(() => window.rustRoboticsEmbedGetState());
    return state.payload.demos[0].n_landmarks;
  }).toBe(12);

  await page.locator("#slam_toolbar_add_demo_button").click();
  await expect(page.locator("#embed_slam_cards .slam-dom-card")).toHaveCount(2);

  await page.locator("#embed_slam_cards .slam-dom-card").nth(1).locator(".card-remove").click();
  await expect(page.locator("#embed_slam_cards .slam-dom-card")).toHaveCount(1);

  const demoCount = await page.locator("#slam_toolbar_count").textContent();
  expect(demoCount).toContain("1 demo");
  expect(pageErrors).toEqual([]);
  expect(consoleErrors).toEqual([]);
});

test("focused robot embed boots with DOM controls and responds to robot selection", async ({
  page,
  baseURL,
}) => {
  const url = `${baseURL}/?mode=robot&embed=focused&ui=smoke_robot`;
  const { consoleErrors, pageErrors } = await bootApp(page, url);

  await page.waitForFunction(() => {
    const state = window.rustRoboticsEmbedGetState?.();
    return state?.mode === "robot" && state?.payload?.kind === "robot";
  });

  await expect(page.locator("#embed_robot_toolbar")).toBeVisible();
  await expect(page.locator("#robot_toolbar_robot_select")).toHaveValue("go2");
  await expect.poll(async () => {
    const state = await page.evaluate(() => window.rustRoboticsEmbedGetState());
    return state.payload.robot.selected_robot;
  }).toBe("go2");

  await page.selectOption("#robot_toolbar_robot_select", "open_duck_mini");
  await expect.poll(async () => {
    const state = await page.evaluate(() => window.rustRoboticsEmbedGetState());
    return state.payload.robot.selected_robot;
  }).toBe("open_duck_mini");

  await page.locator("#robot_toolbar_reset_view_button").click();
  await expect(page.locator("#robot_status_text")).not.toHaveText("");

  expect(pageErrors).toEqual([]);
  expect(consoleErrors).toEqual([]);
});
