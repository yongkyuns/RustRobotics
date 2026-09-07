import { expect, test } from "@playwright/test";

test("focused pendulum embed boots with its DOM teaching surface", async ({ page, baseURL }) => {
  const consoleErrors = [];
  const pageErrors = [];

  page.on("console", (message) => {
    if (message.type() === "error" || message.type() === "assert") {
      consoleErrors.push(message.text());
    }
  });
  page.on("pageerror", (error) => pageErrors.push(error.message));

  const response = await page.goto(
    `${baseURL}/?mode=inverted_pendulum&embed=focused&ui=smoke_pendulum`,
    { waitUntil: "domcontentloaded" },
  );
  expect(response?.ok()).toBeTruthy();

  await page.waitForFunction(() => !document.getElementById("center_text"), null, {
    timeout: 60_000,
  });
  await page.waitForFunction(() => {
    return (
      typeof window.rustRoboticsTestGetState === "function" &&
      typeof window.rustRoboticsEmbedGetState === "function"
    );
  });
  await page.waitForFunction(() => {
    const state = window.rustRoboticsTestGetState();
    return state.mode === "inverted_pendulum";
  });

  await expect(page.locator("#embed_toolbar")).toBeVisible();
  await expect(page.locator(".pendulum-dom-card")).toHaveCount(1);
  await expect(page.locator("#the_canvas_id")).toBeVisible();

  const embedState = await page.evaluate(() => window.rustRoboticsEmbedGetState());
  expect(embedState.mode).toBe("inverted_pendulum");

  expect(pageErrors).toEqual([]);
  expect(consoleErrors).toEqual([]);
});
