# Product grounding: public research behind this prototype

Everything in the app maps to something publicly observable about Careem or
the ride-hailing/super-app category. **No internal or confidential data was
used** — only public reviews, news coverage, and Careem's own publications.

## 1. Pain points → features → interventions

| Publicly-documented pain point | Public evidence | Dataset feature | Intervention tested in the app |
|---|---|---|---|
| Captains cancel after accepting; users wait 25–30 min and get asked the destination | [Trustpilot reviews](https://www.trustpilot.com/review/careem.com), [PissedConsumer](https://careem.pissedconsumer.com/review.html) | `captain_cancellations_90d`, `avg_wait_time_min` | Priority re-matching + auto cancellation compensation; dispatch/ETA pilot |
| Refund & billing frustration: charged for cancelled orders, refunds taking weeks–months, scripted support | [Trustpilot](https://www.trustpilot.com/review/careem.com); Careem's own policy says wallet refunds are instant but card refunds take 3+ business days ([Careem Help](https://help.careem.com/hc/en-us/articles/12203237689747-How-long-does-it-take-to-get-a-refund)) | `refund_delays_90d`, `failed_payments_90d`, `support_tickets_90d` | Instant-refund guarantee + wallet credit; payment auto-retry; service recovery |
| Peak/surge pricing frustration — up to 3× fares during rain in Dubai; frequent peak pricing complaints in KSA | [Lovin Dubai](https://lovin.co/dubai/en/news/careem-uber-taxis-rain), [MENAbytes](https://www.menabytes.com/careem-peak-pricing-saudi-customers-not-happy/) | `surge_exposure_share` | Fare-lock passes / Careem Plus upsell |
| Late food deliveries, orders cancelled after long waits | [Trustpilot](https://www.trustpilot.com/review/careem.com), [JustUseApp reviews](https://justuseapp.com/en/app/592978487/careem-rides-food-delivery/reviews) | `delivery_delays_90d` | Delivery-time guarantee with auto-credit |

## 2. The cross-vertical thesis is Careem's own public strategy

- Careem's Everything App strategy is explicitly about *"increasing the number
  of reasons users open the app each day"* to strengthen retention —
  [Building the Everything App](https://why.careem.com/en/building-the-everything-app/).
- Careem Plus members are publicly reported to show **~3× higher retention and
  use ~2× as many services**; by mid-2025 ~45% of UAE transaction volume came
  from subscribers.
- **Validation:** in our synthetic data, non-subscribers churn at **2.8×** and
  single-vertical users at **3.5×** the rate of others — the same order of
  magnitude as Careem's public numbers, without being fit to them.
- Careem's Data/AI leadership describes ML that *"classifies, personalizes,
  contextualizes, anticipates, recommends"* — e.g. one-click widgets for
  frequent destinations and cuisine-level food personalization
  ([interview with Careem's Selim Turki](https://www.mckinsey.com/capabilities/quantumblack/our-insights/ai-and-the-super-app-an-interview-with-careems-selim-turki)).
  This prototype's what-if simulator and retention playbook are designed as
  inputs to exactly that kind of personalization layer.

## 3. Category benchmarks used for framing

- On-demand apps retain ~20–30% of users long-term; ~71% of new installs never
  return within 90 days ([Sendbird benchmarks](https://sendbird.com/blog/app-retention-benchmarks-broken-down-by-industry),
  [Onde](https://onde.app/blog/app-retention-rate)) — churn management is the
  category's core economics problem, not a side quest.

## 4. What the root-cause analysis adds (and its limits)

The app deliberately shows **three levels of "why"**:

1. **Global model drivers** — permutation importance: what the model relies on.
2. **Observational evidence** — churn rate exposed vs not exposed per pain
   point, with relative lift: why the numbers look the way they do.
3. **Local explanations** — per-user counterfactual attribution: why *this*
   user is at risk, in plain language.

And it is honest about the traps:

- **Confounding** — surge-exposed users show a raw churn lift *below 1* because
  surge exposure correlates with heavy riding (which protects). The app flags
  this explicitly instead of hiding it — raw lifts and model attribution are
  shown side by side.
- **Correlation ≠ causation** — every intervention estimate is labelled a
  model-based counterfactual, and the leadership view ships with an A/B design
  (hypothesis, primary metric, guardrails) for the top plays.

## Sources

- https://www.trustpilot.com/review/careem.com
- https://careem.pissedconsumer.com/review.html
- https://justuseapp.com/en/app/592978487/careem-rides-food-delivery/reviews
- https://help.careem.com/hc/en-us/articles/12203237689747-How-long-does-it-take-to-get-a-refund
- https://lovin.co/dubai/en/news/careem-uber-taxis-rain
- https://www.menabytes.com/careem-peak-pricing-saudi-customers-not-happy/
- https://why.careem.com/en/building-the-everything-app/
- https://www.mckinsey.com/capabilities/quantumblack/our-insights/ai-and-the-super-app-an-interview-with-careems-selim-turki
- https://sendbird.com/blog/app-retention-benchmarks-broken-down-by-industry
- https://onde.app/blog/app-retention-rate
