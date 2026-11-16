import numpy as np

from trustissues.calibrate import robust_line


def test_recovers_the_line_with_forty_percent_liars_and_names_them():
    rng = np.random.default_rng(0)
    raw = rng.uniform(150, 1000, 200)
    truth = 0.0125 * raw - 1.8 + rng.normal(0, 0.03, 200)
    liars = rng.random(200) < 0.4
    truth[liars] += rng.uniform(2, 6, liars.sum())

    fit = robust_line(raw, truth)

    assert abs(fit.slope - 0.0125) < 0.0002
    assert abs(fit.offset + 1.8) < 0.1
    assert not fit.inliers[liars].any()          # no liar believed
    assert fit.inliers[~liars].mean() > 0.95     # nearly every honest reading kept
