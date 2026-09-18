import { expect, test } from "@playwright/test";

function configuration(environmentCount) {
  const config = {
    env: {
      dt: 0.01, max_force: 20,
      reset_position_range_m: 0.2, reset_velocity_range_mps: 0.4,
      reset_angle_range_rad: 0.25, reset_angular_velocity_range_radps: 0.5,
      max_angle_rad: 0.6, max_position_m: 2.4, max_steps: 11,
      observation_position_noise_m: 0.002, observation_velocity_noise_mps: 0.01,
      observation_angle_noise_rad: 0.002, observation_angular_velocity_noise_radps: 0.01,
      action_noise_force_n: 0.15, disturbance_force_n: 1,
      disturbance_probability_per_step: 0.005,
      reward_position_weight: 0.2, reward_velocity_weight: 0.02,
      reward_angle_weight: 1, reward_angular_velocity_weight: 0.05,
      reward_action_weight: 0.001,
    },
    ppo: {
      rollout_steps: 8, mini_batch_size: 8, epochs_per_update: 2,
      gamma: 0.99, gae_lambda: 0.95, clip_epsilon: 0.2,
      value_loss_coef: 0.5, entropy_coef: 0, learning_rate: 0.0003,
    },
    hidden_dim: 8, action_std: 2, sync_policy_each_update: true, seed: 201,
  };
  if (environmentCount !== undefined) config.environment_count = environmentCount;
  return config;
}

async function boot(page) {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const response = await page.goto("/?mode=inverted_pendulum&embed=focused", {
    waitUntil: "domcontentloaded",
  });
  expect(response.ok()).toBeTruthy();
  await page.waitForFunction(() => !document.getElementById("center_text") &&
    typeof globalThis.rustRoboticsPpoTrainerCreate === "function" &&
    typeof globalThis.wasm_bindgen?.rust_robotics_ppo_worker_create_trainer === "function");
  return errors;
}

async function read(page, handle) {
  return page.evaluate((h) => globalThis.rustRoboticsPpoTrainerPoll(h), handle);
}

async function ready(page, handle, updates) {
  await expect.poll(async () => {
    const state = await read(page, handle);
    if (state?.error) throw new Error(state.error);
    return state?.ready && !state.busy && state.metrics?.total_updates === updates;
  }).toBe(true);
  return read(page, handle);
}

async function train(page, handle, count, expectedUpdates) {
  expect(await page.evaluate(([h, n]) =>
    globalThis.rustRoboticsPpoTrainerRequestTrain(h, n), [handle, count])).toBe(true);
  return ready(page, handle, expectedUpdates);
}

for (const count of [1, 3]) {
  test(`real WASM worker preserves pooled optimizer history with ${count} environments`, async ({ page }) => {
    const errors = await boot(page);
    const config = configuration(count);
    const handles = await page.evaluate((cfg) => [
      globalThis.rustRoboticsPpoTrainerCreate(cfg),
      globalThis.rustRoboticsPpoTrainerCreate(cfg),
    ], config);
    try {
      const first = await ready(page, handles[0], 0);
      expect((await ready(page, handles[1], 0)).shared_state).toEqual(first.shared_state);
      const grouped = await train(page, handles[0], 4, 4);
      let split;
      for (let i = 1; i <= 4; i++) split = await train(page, handles[1], 1, i);
      expect(split.shared_state).toEqual(grouped.shared_state);
      expect(split.metrics).toEqual(grouped.metrics);
      expect(split.metrics.total_env_steps).toBe(4 * 8 * count);
      expect(split.metrics.total_episodes).toBeGreaterThanOrEqual(2 * count);
      expect(split.shared_state).not.toEqual(first.shared_state);
      // Compare against the same real learner running directly in this WASM
      // instance, not a mock or a portable inference-only reconstruction.
      const direct = await page.evaluate((cfg) => {
        const api = globalThis.wasm_bindgen;
        const created = api.rust_robotics_ppo_worker_create_trainer(cfg);
        try { return api.rust_robotics_ppo_worker_train(created.session_id, 4); }
        finally { api.rust_robotics_ppo_worker_destroy_trainer(created.session_id); }
      }, config);
      expect(direct.shared_state).toEqual(grouped.shared_state);
      expect(direct.metrics).toEqual(grouped.metrics);
      expect(await read(page, handles[0])).toEqual(grouped);
      expect(errors).toEqual([]);
    } finally {
      await page.evaluate((hs) => hs.forEach((h) => globalThis.rustRoboticsPpoTrainerDestroy(h)), handles);
    }
    expect(await read(page, handles[0])).toBeNull();
  });
}

test("WASM creation keeps legacy flat config and rejects invalid pools without poisoning recovery", async ({ page }) => {
  const errors = await boot(page);
  const result = await page.evaluate((cfg) => {
    const api = globalThis.wasm_bindgen;
    let rejected = false;
    try { api.rust_robotics_ppo_worker_create_trainer({ ...cfg, environment_count: 0 }); }
    catch (error) { rejected = error instanceof Error; }
    const legacy = api.rust_robotics_ppo_worker_create_trainer(cfg);
    const explicit = api.rust_robotics_ppo_worker_create_trainer({ ...cfg, environment_count: 1 });
    try {
      return { rejected, a: api.rust_robotics_ppo_worker_train(legacy.session_id, 2),
        b: api.rust_robotics_ppo_worker_train(explicit.session_id, 2) };
    } finally {
      api.rust_robotics_ppo_worker_destroy_trainer(legacy.session_id);
      api.rust_robotics_ppo_worker_destroy_trainer(explicit.session_id);
    }
  }, configuration());
  expect(result.rejected).toBe(true);
  expect(result.a).toEqual(result.b);
  expect(result.a.metrics.total_env_steps).toBe(16);
  expect(errors).toEqual([]);
});

test("browser coordinator reports pooled environments and one coherent update budget", async ({ page }) => {
  const errors = await boot(page);
  await page.waitForFunction(() => globalThis.rustRoboticsEmbedGetState?.().payload?.pendulums?.length > 0);
  const id = await page.evaluate(() => globalThis.rustRoboticsEmbedGetState().payload.pendulums[0].id);
  await page.evaluate((id) => {
    globalThis.wasm_bindgen.rust_robotics_test_patch_pendulum(id, JSON.stringify({
      policy: { parallel_trainers: 3, rollout_steps: 32, epochs_per_update: 1, training_updates_per_tick: 1 },
      trainer_action: "start",
    }));
  }, id);
  await expect.poll(async () => page.evaluate((id) => {
    const p = globalThis.rustRoboticsEmbedGetState().payload.pendulums.find((p) => p.id === id);
    if (p.policy_trainer.last_error) throw new Error(p.policy_trainer.last_error);
    return p.policy_trainer.metrics?.total_updates ?? 0;
  }, id)).toBeGreaterThanOrEqual(2);
  await page.evaluate((id) => globalThis.wasm_bindgen.rust_robotics_test_patch_pendulum(
    id, JSON.stringify({ trainer_action: "stop" })), id);
  await expect.poll(async () => page.evaluate((id) => {
    const t = globalThis.rustRoboticsEmbedGetState().payload.pendulums.find((p) => p.id === id).policy_trainer;
    return !t.training_active && !t.busy && t.ready_replicas === 3;
  }, id)).toBe(true);
  const state = await page.evaluate((id) => globalThis.rustRoboticsEmbedGetState().payload.pendulums
    .find((p) => p.id === id).policy_trainer, id);
  expect(state.total_replicas).toBe(3);
  expect(state.metrics.total_env_steps).toBe(state.metrics.total_updates * 3 * 32);
  expect(state.snapshot_ready).toBe(true);
  expect(errors).toEqual([]);
});
