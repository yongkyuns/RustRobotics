# Browser-tooling dependency security

## Scope and audit policy

The root `package.json` is a private browser-test package. Its only direct npm
dependency is the development dependency `@playwright/test`. These packages run
on developer machines and CI workers; `build_web.sh` does not copy them into the
simulator's static bundle.

Run the same audit used by the **npm dependency audit** CI check:

```sh
npm ci --include=dev --ignore-scripts --no-audit
npm run audit:npm
```

The command explicitly includes development dependencies and fails for low,
moderate, high, or critical findings (`--audit-level=low`). Registry/network
errors also fail the check. CI preserves the JSON report as an artifact, and
its Bash pipeline uses `pipefail` so saving the report cannot hide audit failure.
The audit-only installation skips lifecycle scripts; the WASM/browser job still
performs its normal `npm ci` and explicit matching Chromium installation.

There are no accepted findings or suppressions. A new finding must be
investigated and repaired before merge. Any proposed exception requires a
separate reviewed policy change documenting the exact advisory and dependency
path, reachability, mitigation, owner, tracking issue, and expiry date. Do not
omit development dependencies, raise the severity threshold, add
`continue-on-error`, or use `npm audit fix --force` to make the check pass.

This audit covers the root npm lockfile, not separately vendored MuJoCo/ONNX
assets, browser/system packages, Rust dependencies, or the deployed site's
complete attack surface. A clean `npm audit --omit=dev` is diagnostic only and
is not an acceptable replacement for the full tooling audit.

## Issue #22: verified finding and repair

The investigation ran on 2026-09-08 (UTC), with Node 22.23.2 and npm 10.9.8,
against the unchanged root npm manifests from master `b2f1791`:

- `npm audit --include=dev --json` reported two high-severity affected packages,
  but only **one underlying advisory**: **GHSA-7mvr-c777-76hp / CVE-2025-59288**.
- `playwright@1.54.2` carried the advisory; `@playwright/test@1.54.2` was reported
  transitively through its exact dependency on `playwright`. The dependency path
  was `@playwright/test -> playwright -> playwright-core`, all at 1.54.2.
  `playwright-core` contains the installer implementation but was not a separate
  affected-package entry in this npm report.
- `npm audit --omit=dev --json` reported zero findings. This does not establish
  that the full deployed simulator is vulnerability-free.

The advisory concerns macOS Chrome/Edge installer scripts that used `curl -k`,
allowing an intercepted download to substitute an executable installer. Upstream
lists versions below 1.55.1 as affected and 1.55.1 as the first patched version.
The repository's browser CI installs Chromium on Ubuntu, not those macOS branded
browser scripts. The finding therefore does not establish exploitation of that
CI path or an MPC/browser-runtime vulnerability. Developer tooling still needs
the dependency repair.

The repair pins `@playwright/test`, `playwright`, and `playwright-core` to 1.56.1,
using an npm-generated lockfile. This is a targeted fixed-version update, not a
claim that 1.56.1 is the latest release. The optional `fsevents@2.3.2` dependency
is unchanged. The regenerated tree passed a fresh `npm ci` and reported zero
findings with development dependencies included. All nine existing browser tests
were discovered without changing their assertions or configuration.

Evidence and upstream references:

- [Issue #22](https://github.com/yongkyuns/RustRobotics/issues/22)
- [Investigation log and before/after JSON artifacts](https://github.com/yongkyuns/RustRobotics/actions/runs/34172903001)
- [GitHub advisory](https://github.com/advisories/GHSA-7mvr-c777-76hp)
- [Upstream installer fix](https://github.com/microsoft/playwright/commit/72c62d840247d9defd87c6beb0344d456794b570)
- [Playwright 1.56.1 release](https://github.com/microsoft/playwright/releases/tag/v1.56.1)

## Qualifying a dependency update

Use Node 22, preserve the exact version pin, and let npm regenerate the lockfile:

```sh
npm install --save-dev --save-exact @playwright/test@<reviewed-version>
npm ci
npm run audit:npm
npm ls --all
```

Then, with the Rust/WASM prerequisites from `.github/workflows/rust.yml` installed:

```sh
./build_web.sh --fast
npx --no-install playwright install --with-deps chromium
npm run test:web-smoke
```

Playwright versions require matching browser binaries; do not reuse an old
Chromium executable to qualify a new lockfile. The complete suite includes the
interactive MPC weight-change, LQR/MPC switch, and restart regression. Test
listing alone is not browser execution. Require all browser and normal PR checks
to pass on the final revision before merge, without weakening existing tests.
