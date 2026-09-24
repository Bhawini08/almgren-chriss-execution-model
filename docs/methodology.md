# Methodology

The implementation uses the standard Almgren-Chriss mean-variance liquidation objective. A deterministic inventory path trades off temporary impact against variance from carrying inventory through a volatile market. Permanent impact contributes to expected shortfall but does not change the basic continuous-time trajectory under the simplified linear-impact specification used here.

For zero risk aversion, the schedule converges to uniform liquidation. Increasing risk aversion increases the urgency parameter and front-loads execution, reducing cost variance while typically increasing temporary-impact cost.

The implementation-shortfall simulation adds Gaussian price shocks to the deterministic impact cost so analytical cost variance can be compared with Monte Carlo dispersion. The current parameter set is stylized and should not be interpreted as calibrated market impact.
