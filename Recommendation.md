# Recommendation — Washing-machine insight from smart-meter data

*For: utility customer-programme and analytics teams. Based on 20 UK homes (REFIT, 2013–15);
model tested on two homes it had never seen.*

## What the model can and cannot do

| Suitable uses | Not suitable (yet) |
|---|---|
| Flagging **whether and when** a household runs hot washes (finds ~75 % of washes in unseen homes) | Itemised bills or anything with financial consequences per customer |
| Aggregate statistics over many homes (e.g. share of washing in the evening peak) | Exact kWh per customer: energy error was −35 % in one test home and +92 % in the other |
| Personalised tips, with wording that allows for error ("it looks like…") | Real-time alerts: accuracy drops when only past data can be used |
| Prioritising homes for a programme, then verifying with a cheap smart plug | 15/30-minute meter data (see below) or non-UK appliance stock without re-validation |

## Observed findings (measured on 9 sub-metered homes, 3,419 washes)

* **Water heating is ~80 % of washing-machine energy** (70–90 % per home): a ~2 kW heater runs
  for 10–20 minutes at the start of each hot wash. Energy per wash varies 2.6-fold between homes
  (median 0.31–0.81 kWh), mostly through programme/temperature choice, not frequency.
* **16 % of washes start in the 16:00–19:00 evening peak**, between 0.6 % and 31 % depending on the
  home. One home already runs 76 % of its washes between 01:00 and 07:00 (a delay timer), so
  shifting is practical. Washing is otherwise a morning, Saturday-heavy activity.
* Washing machines are 1–7 % of these homes' electricity (≈40–330 kWh/year).

## Proposed opportunity: "Your washing, cooler and later"

**Customer problem.** People do not know what a hot wash costs, or that the default evening
wash lands on the most expensive/constrained hours.

**Intervention.** For customers whose meter data shows frequent heating-phase washes and/or
evening-peak washes, add one insight card to the app/bill: estimated washes per week, share
during 16:00–19:00, and two actions — choose 30 °C/eco programmes, and use the delay-start
timer (paired with a time-of-use tariff where offered). Use the model to **target** customers
and to **track** behaviour, not to bill them.

**How we would know it worked (not assumed).**
1. Randomised pilot: eligible customers split into treatment (insight card) and control (no
   card), stratified by baseline usage; pre-registered outcomes; 8–12 weeks.
2. Give a random ~10 % sub-sample of both arms a smart plug on the washing machine. This
   gives ground truth for the outcomes **and** re-validates the model on the local population.
3. Primary outcomes: share of washes started 16:00–19:00 and kWh per wash (from the plugged
   sub-sample), compared treatment vs control (difference-in-differences against the baseline
   period). Secondary: total household kWh and peak-period kWh from the meter; app engagement.
4. Success = a statistically significant reduction in peak-share or kWh/wash versus control,
   with no rise in complaints about inaccurate insights. We make no savings claim before then.

## Limitations

* **Overlapping signatures**: kettles, dishwashers, tumble dryers, washer-dryers, showers and
  immersion heaters also make 2–3 kW steps. The worst test errors came from exactly this.
* **Missing data**: 6–25 % of minutes were missing per home, with outages of up to 62 days;
  the model can neither see nor report washes during gaps.
* **Household differences**: accuracy varied strongly between the two test homes; a model that
  had already seen a home was much better (half the power error), which argues for a short
  per-customer calibration step.
* **Small sample**: 20 UK homes, one region, 2013–15 appliances. Two test homes give no
  reliable confidence interval.

## Applying UK results to India

The model's strongest cue — a ~2 kW internal heater at the start of a front-loader cycle —
is often **absent** in India, where semi-automatic and top-loading machines washing in cold
water are common. Meanwhile **other ~2 kW loads are common** (water geysers, irons, induction
hobs, air-conditioner compressors). Behaviour also differs: washing tied to water-supply
hours, larger households, shared/joint-family meters, and strong seasonality (summer AC,
monsoon). Electrical conditions differ too: voltage fluctuations, frequent outages, and
inverter/battery back-up or rooftop solar behind the meter, all of which distort the aggregate.
**Before use in India**: a local sub-metered pilot (smart plugs on washing machines in a few
hundred homes across climate zones and tariff types), retraining or re-thresholding on that
data, separate validation per segment, and a check that the meter data resolution supports it.

## If only 15- or 30-minute smart-meter data were available

A washing machine's identifiable feature — a 10–20-minute 2 kW heating block followed by an
hour of 50–150 W tumbling — is shorter than or similar to a single 15/30-minute interval. At
that resolution the heating phase becomes one modest bump in interval energy (≈0.3–0.6 kWh),
indistinguishable from a kettle plus cooking in the same interval; the tumbling and spin phases,
the start time (to within 15–30 min) and back-to-back cycles disappear; concurrent loads are
summed together; and there are 15–30× fewer data points to learn from. Cycle detection and
minute-level power estimates would no longer be realistic. What remains feasible is coarser:
estimating washing energy over weeks or months statistically, detecting "probably washed
today", and segmenting customers — and even that would need to be re-validated, because this
project only measured performance at 1-minute resolution.
