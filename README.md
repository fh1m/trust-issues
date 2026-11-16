# trust-issues

**A straight line, a depth sensor, and the fraction of liars it can survive.**

My first machine-learning script (November 2023) fit a line through
100 polite, made-up points with ordinary least squares. Least squares is a
wonderful listener. It weighs every point equally, including the ones that
are lying.

<img src="assets/original/ols-2023.png" width="420" alt="The 2023 plot: a blue least-squares line through twenty test points">

*November 2023. Top right: one point near y ≈ 11 sits about two units above
the line. In this script that's honest Gaussian noise. On a real sensor it's
exactly what a liar looks like, and the line would lean towards it all the same.*

On a robot, points lie all the time. The pressure sensor that tells an
underwater vehicle its depth is calibrated by fitting exactly this kind of
line, raw counts → metres, and the calibration data comes with bubbles on
the port, cable tugs and the occasional bit-flip on the bus. So the question
became: **what fraction of the data can lie before the depth estimate is
useless?** Statisticians call that the *breakdown point*.

![Median depth error against the fraction of lying readings, four estimators](assets/breakdown.png)

## The interrogation

Each trial is a 120-reading calibration run. A growing fraction of readings
gets a spike of +2 to +6 m. Four estimators fit the line; the score is the
mean depth error across the sensor's working range, median over 40 trials.

| liars | least squares | Huber | Theil–Sen | RANSAC |
|---:|---:|---:|---:|---:|
| 0 % | 0.3 cm | 0.3 cm | 0.5 cm | 0.3 cm |
| 5 % | 22.7 cm | 0.4 cm | 0.6 cm | 0.4 cm |
| 20 % | 84.7 cm | 1.2 cm | 0.7 cm | 0.3 cm |
| 35 % | 137.9 cm | 8.6 cm | 1.2 cm | 0.3 cm |
| 50 % | 199.5 cm | 185.7 cm | 8.2 cm | 0.4 cm |
| 60 % | 244.3 cm | 239.3 cm | 292.3 cm | 0.5 cm |
| **under 5 cm until** | **0 %** | **30 %** | **45 %** | **60 %+** |

Here is one of those runs at 30 % liars. Least squares splits the difference
between the truth and the lies; the other three ignore the shouting.

![A single calibration run at 30 % liars with all four fitted lines](assets/thirty_percent.png)

## Verdicts

- **Least squares** breaks with *one* liar; its breakdown point is 0 %.
  Five percent of bad readings put the vehicle 23 cm off in depth.
- **Huber** bends instead of breaking, and is cheap enough for a
  microcontroller. It collapses abruptly between 35 and 40 %.
- **Theil–Sen** takes the median of pairwise slopes. Its theoretical
  breakdown is ~29 % but it holds longer here, because these liars scatter
  rather than agree.
- **RANSAC** wins this round, and the reason is worth knowing: it trusts the
  largest group that *agrees*. These liars disagree with each other, so they
  never outvote the truth. Liars that conspire (a sensor that drifts
  consistently) would beat it. No estimator survives an organised lie. That's
  what a second, independent sensor is for.

## Use it on your own sensor

The study became a tool. Give it a CSV of raw readings against ground
truth and it fits a calibration that doesn't take the liars' side:

```bash
python -m trustissues.calibrate examples/synthetic_calibration.csv --plot fit.png
```
```
truth = 0.0124264 * raw + -1.71894
believed 46 of 60 readings
  refused row 13: raw=589 truth=9.347
  refused row 17: raw=963 truth=15.739
  …
plain least squares would be off by up to 2.233 truth units across this range
```

![The calibration tool on the example file](assets/calibrate_example.png)

RANSAC decides *who* to believe, then plain least squares refits the
believers. Least squares is still the best judge of honest witnesses. The
inlier band defaults to 2.5 robust standard deviations of a rough fit, so
it works in any unit. The example file is synthetic: a made-up sensor with a
quarter of its readings spiked.

## Run it

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m trustissues.run     # ~50 s, prints the table, rewrites assets/*.png
pytest -q                     # 40 % liars in, the right line and the right suspects out
```

Seeded and deterministic.

## Layout

```
origin/Main.py           the 2023 script: OLS on 100 well-behaved points
README-2023.md           its README and screenshot
trustissues/run.py       the sweep, the trial, both figures
trustissues/calibrate.py the tool
```
