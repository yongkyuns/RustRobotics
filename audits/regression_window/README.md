# Historical regression-window replay — complete

[Results and limitations](RESULTS.md)

Executed source **a3276e1efa3071d381480bc61f1abe1ebd818a54**, run **35554192268**, branch `audit/ppo-regression-window-20260920`. Registration: issue #35 comment **5754579087**, before new outcomes. All three jobs completed successfully on the first attempt at September 21, 2026, 02:54:38 UTC. This replaces the earlier in-progress status; no duplicate training was launched during verification.

## Finding, not a new controller

Seed **41006/update 4,140** reduced independent outward-recovery completions from **208/256 to 196/256**. The adjusted 99% paired completion-effect interval is **[−9.1638, −0.2112] percentage points**. Nominal outcome identities were unchanged on those draws; that is not a safety certificate. The same update improved the fitted clipped surrogate by **+0.008415** and reduced critic MSE against its sampled targets.

The selected **41004/update 4,137** also reduced common-discount return on all three independent panels; its completion differences remain inconclusive. A history that ultimately improves can still contain locally harmful updates.

The full 96-update chronology includes rebounds and contrary results. Neither selected transaction explains the entire training trajectory, and no particular critic estimate or advantage component has been isolated as the cause. No correction, rollback, production checkpoint selection, merge or deployment occurred.

## Registered design

Reconstruct the actual support1024/gamma0.995 histories of exposed seeds41006 and41004 from their original initialization through4224 with live Adam, environment, unfinished episode and random state. Require every update record and all historical actor/critic checkpoint comparisons to match exactly. Historical files are comparison inputs, not warm starts or portable optimizer-resume files.

Record every update4129–4224, including all fitting/return-support data and16 optimizer transactions. Evaluate all97 snapshots on the same three historical20.48-second panels with64 cases each. Select the largest consecutive outward-completion decrease per history, earliest tie, only for diagnosis. Then test its frozen before/after actors on256 independent paired cases per panel. Evaluation never enters training or update selection.

The fixed interval family is12 contrasts:two selected histories,three panels,completion and common-gamma.99 discounted return. These conditional intervals are not training-population or hardware qualification. Training gamma remains.995 within this experimental recipe; the common evaluation return is not an unbiased estimate of the training-discount expected objective.

## Verification

Native preflight:111 unit/audit tests,7 ordinary balancing and9 ordinary learning controls,Clippy,formatting and protected-source restoration. The current endpoint was explicitly run after regular preflight; ignored historical heavy endpoints and the production browser/platform matrix were not rerun.

The unchanged109-test offline suite passes on actual results. All three current raw ZIP digests and15165 payload hashes,plus two nested historical archives/548 payloads,verify. Both full prefix histories and window endpoints reproduce. All40320 evaluation records,192 detailed updates,3072 optimizer transactions and24 exported full evaluation paths/40327 transitions are checked. Returns and normalization reproduce bit-for-bit. Unexported native gradients,Adam moments,RNG draws and other full evaluation paths are not independently regenerated. Logged nonsmooth finite-difference residuals and all contrary findings are retained in the full report.

The conversation evidence provides unchanged raw archives,compiled source,all numerical reports,offline code/tests and segmented reproduction instructions. Offline verification does not run the native binary,network,simulator or learning. A green verifier is not a controller-qualification pass.

## Disposition

The longer-support recipe remains rejected. Production gamma0.99/lambda0.95,rewards,normalization,learned policies,PR38,master,deployment and fallback behavior remain unchanged. The new result is an identified, independently confirmed harmful update suitable for a focused counterfactual correction test; no such correction has yet been tested.