/* Churn Lab — product tool frontend: run registry, role views, workspaces. */

const $ = (sel) => document.querySelector(sel);
const fmtPct = (x, d = 1) => (x * 100).toFixed(d) + "%";
const fmtAED = (x) => "AED " + Math.round(x).toLocaleString("en-US");
const fmtNum = (x) => Math.round(x).toLocaleString("en-US");

let D = null; // current run bootstrap

/* ---------------- roles & views ---------------- */
const VIEWS = {
  experiments: { title: "Experiments", sub: "Configure and run the autonomous loop; every run lands in the registry." },
  rootcause:   { title: "Root causes", sub: "Three levels of why: model drivers, pain-point evidence, per-user explanations." },
  playbook:    { title: "Retention playbook", sub: "Who to save, how, and what it's worth — ranked by revenue protected." },
  whatif:      { title: "What-if studio", sub: "Counterfactual simulation of interventions on the current run's holdout." },
  leadership:  { title: "Leadership view", sub: "The decision, not the dashboard: what's at stake, what to do, how we'd test it." },
  narrator:    { title: "Experiment Narrator", sub: "The metrics JSON becomes a leadership brief — live via Claude or deterministic template." },
  method:      { title: "Methodology", sub: "How every number is produced — pipeline, definitions, limitations, and the path to production." },
};
const ROLES = {
  ds:   { label: "Data Scientist", views: ["experiments", "rootcause", "playbook", "whatif", "leadership", "narrator", "method"], landing: "experiments" },
  pm:   { label: "Product",        views: ["rootcause", "playbook", "whatif", "narrator", "method"], landing: "playbook" },
  exec: { label: "Executive",      views: ["leadership", "narrator"], landing: "leadership" },
};
let role = "ds";

function showView(name) {
  document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === "view-" + name));
  document.querySelectorAll(".nav-item").forEach((n) => n.classList.toggle("active", n.dataset.view === name));
  $("#view-title").textContent = VIEWS[name].title;
  $("#view-sub").textContent = VIEWS[name].sub;
  window.scrollTo({ top: 0 });
}

function applyRole(r) {
  role = r;
  document.querySelectorAll("#role-switch button").forEach((b) => b.classList.toggle("active", b.dataset.role === r));
  document.querySelectorAll(".nav-item").forEach((n) =>
    n.classList.toggle("hidden", !ROLES[r].views.includes(n.dataset.view)));
  showView(ROLES[r].landing);
}

$("#nav").addEventListener("click", (e) => {
  const item = e.target.closest(".nav-item");
  if (item) showView(item.dataset.view);
});
$("#role-switch").addEventListener("click", (e) => {
  const btn = e.target.closest("button");
  if (btn) applyRole(btn.dataset.role);
});

/* ---------------- tiny renderers ---------------- */
function mdToHtml(md) {
  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const inline = (s) =>
    esc(s).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/`(.+?)`/g, "<code>$1</code>");
  let html = "", list = null;
  const closeList = () => { if (list) { html += `</${list}>`; list = null; } };
  for (const raw of md.split("\n")) {
    const line = raw.trim();
    if (!line) { closeList(); continue; }
    if (/^- /.test(line)) {
      if (list !== "ul") { closeList(); html += "<ul>"; list = "ul"; }
      html += `<li>${inline(line.slice(2))}</li>`;
    } else if (/^\d+\. /.test(line)) {
      if (list !== "ol") { closeList(); html += "<ol>"; list = "ol"; }
      html += `<li>${inline(line.replace(/^\d+\. /, ""))}</li>`;
    } else { closeList(); html += `<p>${inline(line)}</p>`; }
  }
  closeList();
  return html;
}

const statTile = (label, value, note = "") =>
  `<div class="stat"><div class="stat-label">${label}</div>
   <div class="stat-value">${value}</div>${note ? `<div class="stat-note">${note}</div>` : ""}</div>`;

function hBars(rows, { max = null, fmt = (v) => v.toFixed(3) } = {}) {
  const m = max ?? Math.max(...rows.map((r) => r.value));
  return rows.map((r) => `
    <div class="bar-row" title="${r.label}: ${fmt(r.value)}">
      <div class="bar-label">${r.label}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${(100 * r.value / m).toFixed(1)}%"></div></div>
      <div class="bar-value">${fmt(r.value)}</div>
    </div>`).join("");
}

function divergingBars(rows, fmt) {
  const m = Math.max(...rows.map((r) => Math.abs(r.value)));
  return rows.map((r) => {
    const w = (50 * Math.abs(r.value) / m).toFixed(1);
    return `
    <div class="div-row" title="${r.label}: ${fmt(r.value)}">
      <div class="bar-label">${r.label}</div>
      <div class="div-track"><div class="div-mid"></div><div class="div-fill ${r.value >= 0 ? "pos" : "neg"}" style="width:${w}%"></div></div>
      <div class="bar-value">${fmt(r.value)}</div>
    </div>`;
  }).join("");
}

function table(headers, rows, { winnerIdx = -1, rowAttrs = () => "" } = {}) {
  const head = headers.map((h) => `<th class="${h.num ? "num" : ""}">${h.label}</th>`).join("");
  const body = rows.map((cells, i) =>
    `<tr class="${i === winnerIdx ? "winner" : ""}" ${rowAttrs(i)}>` +
    cells.map((c, j) => `<td class="${headers[j].num ? "num" : ""}">${c}</td>`).join("") + "</tr>").join("");
  return `<div class="table-scroll"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

function setupTabs() {
  document.querySelectorAll(".tabs").forEach((bar) => {
    bar.querySelectorAll(".tab").forEach((btn) => {
      btn.addEventListener("click", () => {
        bar.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        bar.parentElement.querySelectorAll(":scope > .tabpanel").forEach((p) =>
          p.classList.toggle("active", p.id === btn.dataset.tab));
      });
    });
  });
}

/* ---------------- run registry ---------------- */
async function loadRunHistory() {
  const runs = await (await fetch("/api/runs")).json();
  $("#run-history").innerHTML = table(
    [{ label: "Run" }, { label: "Time" }, { label: "Users", num: 1 }, { label: "Seed", num: 1 },
     { label: "Best model" }, { label: "AUC", num: 1 }, { label: "Revenue at risk", num: 1 }],
    runs.map((r) => [r.run_id, r.created_at, fmtNum(r.n_users), r.seed, r.best_model,
      r.test_auc.toFixed(3), fmtAED(r.revenue_at_risk)]),
    { rowAttrs: (i) => `class="clickable ${runs[i].run_id === (D && D.run.id) ? "current" : ""}" data-run="${runs[i].run_id}"` });
  $("#run-history").querySelectorAll("tr[data-run]").forEach((tr) =>
    tr.addEventListener("click", async () => {
      const b = await (await fetch("/api/bootstrap?run_id=" + tr.dataset.run)).json();
      render(b); loadRunHistory();
    }));
}

async function executeRun() {
  const btn = $("#run-btn");
  const models = [...document.querySelectorAll("#model-checks input:checked")].map((c) => c.value);
  btn.disabled = true; btn.textContent = "Running autonomous loop…";
  $("#status-chip").textContent = "● running"; $("#status-chip").classList.add("busy");
  const t0 = Date.now();
  try {
    const res = await fetch("/api/run", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        n_users: +$("#in-users").value, seed: +$("#in-seed").value,
        test_size: +$("#in-test").value, models,
      }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.status);
    render(await res.json());
    $("#run-note").textContent = `Completed in ${((Date.now() - t0) / 1000).toFixed(1)}s — loaded into every workspace.`;
  } catch (e) {
    $("#run-note").textContent = "Run failed: " + e.message;
  } finally {
    btn.disabled = false; btn.textContent = "▶ Run autonomous loop";
    $("#status-chip").textContent = "● ready"; $("#status-chip").classList.remove("busy");
    loadRunHistory();
  }
}

/* ---------------- main render (current run -> all workspaces) ---------------- */
function render(data) {
  D = data;
  const best = D.leaderboard[0];
  $("#run-chip").textContent = `${D.run.id} · ${fmtNum(D.meta.n_users)} users · seed ${D.meta.seed}`;

  /* experiments */
  $("#model-checks").innerHTML = D.available_models.map((m) =>
    `<label><input type="checkbox" value="${m}" ${D.run.params.models.includes(m) ? "checked" : ""}> ${m}</label>`).join("");
  $("#exp-kpis").innerHTML =
    statTile("Churn rate", fmtPct(D.meta.churn_rate), `${fmtNum(D.meta.n_users)} users`) +
    statTile("Best model", D.best_model, `holdout AUC ${best.test_auc.toFixed(3)}`) +
    statTile("Recall @ top 10%", fmtPct(best.recall_at_top10pct, 0), "churners caught in riskiest decile") +
    statTile("Revenue at risk", fmtAED(D.leadership.annual_revenue_at_risk_aed), `${fmtNum(D.meta.n_holdout)}-user holdout / yr`);
  $("#lb-run-tag").textContent = D.run.id;
  $("#leaderboard").innerHTML = table(
    [{ label: "Model" }, { label: "CV AUC", num: 1 }, { label: "Test AUC", num: 1 },
     { label: "PR-AUC", num: 1 }, { label: "Recall @ 10%", num: 1 }, { label: "Brier", num: 1 }, { label: "Train", num: 1 }],
    D.leaderboard.map((r) => [r.model, r.cv_auc_mean.toFixed(3), r.test_auc.toFixed(3),
      r.test_pr_auc.toFixed(3), fmtPct(r.recall_at_top10pct, 0), r.brier_score.toFixed(3),
      r.train_seconds.toFixed(1) + "s"]),
    { winnerIdx: 0 });
  $("#winner-callout").innerHTML =
    `<strong>Winner: ${D.best_model}</strong> — catches ${fmtPct(best.recall_at_top10pct, 0)} of churners
     in the riskiest 10% of users, at a ${fmtPct(D.meta.churn_rate)} base churn rate.`;
  $("#verticals-chart").innerHTML = hBars(
    Object.entries(D.churn_by_verticals).map(([k, v]) => ({ label: `${k} vertical${k === "1" ? "" : "s"}`, value: v })),
    { fmt: (v) => fmtPct(v, 1) });

  /* root causes */
  $("#importance-chart").innerHTML = hBars(
    D.importance.filter((r) => r.importance > 0).map((r) => ({ label: r.feature, value: r.importance })));
  $("#rca-chart").innerHTML = hBars(
    D.rca.map((r) => ({ label: r.segment, value: r.churn_lift })), { fmt: (v) => v.toFixed(1) + "×" });
  const nonSub = D.rca.find((r) => r.segment === "Not a subscriber");
  $("#validation-callout").innerHTML =
    `<strong>External validation:</strong> non-subscribers churn at ${nonSub.churn_lift.toFixed(1)}×
     (${fmtPct(nonSub.churn_if_exposed)} vs ${fmtPct(nonSub.churn_if_not)}) — strikingly consistent
     with Careem's publicly reported ~3× retention for Plus members.`;
  const surge = D.rca.find((r) => r.segment.startsWith("Surge"));
  $("#confounding-callout").innerHTML = surge && surge.churn_lift < 1 ?
    `<strong>Confounding alert:</strong> surge-exposed users show a lift <em>below</em> 1
     (${surge.churn_lift.toFixed(1)}×) — not because surge helps retention, but because surge exposure
     correlates with heavy riding, which protects. The model's driver ranking isolates the marginal
     effect; an A/B test is the only clean answer.` :
    `<strong>Reading lifts:</strong> these are observational comparisons — use them with the model's
     driver ranking, and confirm causality with A/B tests.`;

  const chips = D.explanations.map((e, i) =>
    `<button class="user-chip ${i === 0 ? "active" : ""}" data-i="${i}">${e.user} · ${fmtPct(e.risk, 0)}</button>`).join("");
  $("#user-picker").innerHTML = chips;
  const showUser = (i) => {
    const e = D.explanations[i];
    const top = e.contributions.slice(0, 3).map((c) =>
      `<li><strong>${c.feature}</strong> = ${c.user_value} (typical ${c.typical_value}) adds
       <strong>${(c.risk_contribution >= 0 ? "+" : "") + fmtPct(c.risk_contribution, 0)}</strong></li>`).join("");
    $("#user-summary").innerHTML =
      `<div class="risk-hero">${fmtPct(e.risk, 0)}<small>predicted churn risk · dominant driver: ${e.driver}</small></div>
       <p class="card-sub" style="margin-top:10px">Main reasons vs a typical user:</p><ul>${top}</ul>`;
    $("#contrib-chart").innerHTML = divergingBars(
      e.contributions.map((c) => ({ label: c.feature, value: c.risk_contribution })),
      (v) => (v >= 0 ? "+" : "") + fmtPct(v, 0));
  };
  showUser(0);
  $("#user-picker").onclick = (ev) => {
    const chip = ev.target.closest(".user-chip");
    if (!chip) return;
    document.querySelectorAll(".user-chip").forEach((c) => c.classList.remove("active"));
    chip.classList.add("active"); showUser(+chip.dataset.i);
  };

  /* playbook */
  $("#segments-table").innerHTML = table(
    [{ label: "Dominant driver" }, { label: "Users", num: 1 }, { label: "Avg risk", num: 1 },
     { label: "Recommended action" }, { label: "Assumed save", num: 1 }, { label: "Revenue protected / yr", num: 1 }],
    D.segments.map((s) => [s.driver, s.users, fmtPct(s.avg_risk, 0), s.recommended_action,
      fmtPct(s.assumed_save_rate, 0), fmtAED(s.annual_revenue_protected_aed)]));

  /* what-if */
  $("#whatif-pills").innerHTML = D.what_ifs.map((w, i) =>
    `<button class="pill ${i === 0 ? "active" : ""}" data-i="${i}">${w.intervention}</button>`).join("");
  const showWhatIf = (i) => {
    const w = D.what_ifs[i];
    const delta = w.new_churn_rate - w.baseline_churn_rate;
    $("#whatif-stats").innerHTML =
      statTile("Users affected", fmtNum(w.users_affected)) +
      statTile("Predicted churn rate", fmtPct(w.new_churn_rate),
        `<span class="up">${(delta * 100).toFixed(2)} pts vs ${fmtPct(w.baseline_churn_rate)}</span>`) +
      statTile("Churners prevented", fmtNum(w.churners_prevented)) +
      statTile("Revenue protected / yr", fmtAED(w.annual_revenue_protected_aed));
    $("#whatif-design").innerHTML = w.design ?
      `<h3>How we'd test it</h3>
       <p><strong>Hypothesis:</strong> ${w.design.hypothesis}.</p>
       <p><strong>Primary metric:</strong> ${w.design.primary_metric}.
          &nbsp;<strong>Guardrails:</strong> ${w.design.guardrail}.</p>
       <p class="fineprint">50/50 randomised among affected users, 4 weeks, sized on the simulated effect.</p>` : "";
  };
  showWhatIf(0);
  $("#whatif-pills").onclick = (ev) => {
    const pill = ev.target.closest(".pill");
    if (!pill) return;
    document.querySelectorAll(".pill").forEach((p) => p.classList.remove("active"));
    pill.classList.add("active"); showWhatIf(+pill.dataset.i);
  };

  /* leadership */
  $("#leadership-kpis").innerHTML =
    statTile("Customers at risk (top decile)", fmtNum(D.leadership.customers_at_risk)) +
    statTile("Annual revenue at risk", fmtAED(D.leadership.annual_revenue_at_risk_aed), "holdout only") +
    statTile("#1 churn driver", D.leadership.top_driver) +
    statTile("Best intervention ROI", fmtAED(D.leadership.best_play_revenue) + "/yr");
  $("#whatif-table").innerHTML = table(
    [{ label: "Intervention" }, { label: "Users affected", num: 1 }, { label: "Churn: before → after", num: 1 },
     { label: "Churners prevented", num: 1 }, { label: "Revenue protected / yr", num: 1 }],
    D.what_ifs.map((w) => [w.intervention, fmtNum(w.users_affected),
      `${fmtPct(w.baseline_churn_rate)} → ${fmtPct(w.new_churn_rate)}`,
      fmtNum(w.churners_prevented), fmtAED(w.annual_revenue_protected_aed)]),
    { winnerIdx: 0 });
  const [p1, p2] = D.what_ifs;
  $("#the-ask").innerHTML =
    `<strong>The ask:</strong> approve 4-week A/B pilots for the top two plays —
     <strong>${p1.intervention}</strong> and <strong>${p2.intervention}</strong> — with a combined
     simulated upside of <strong>${fmtAED(D.leadership.top2_combined_revenue)}/year</strong> on the
     holdout alone. Counterfactuals are model-based estimates; the pilots are how we make them real.
     <details class="ab"><summary>A/B designs for both plays</summary><ul>
     ${[p1, p2].map((p) => p.design ? `<li><strong>${p.intervention}</strong> — ${p.design.hypothesis}.
        Primary: ${p.design.primary_metric}. Guardrails: ${p.design.guardrail}.</li>` : "").join("")}
     </ul></details>`;

  /* narrator */
  $("#sample-brief").innerHTML = mdToHtml(D.narrative.sample_llm_brief);
  $("#prompt-text").textContent = D.narrative.prompt;
  $("#narrate-source").textContent = "";
  $("#narrate-output").innerHTML = "";
}

/* ---------------- methodology (static, rendered once) ---------------- */
function renderMethodology() {
  const steps = [
    ["Simulate", "generate_users(n, seed): 16-feature behavioural dataset; churn drawn from a latent logit encoding publicly-documented pain points."],
    ["Split", "Stratified train/holdout split (configurable %) so every metric below is out-of-sample."],
    ["Cross-validate", "5-fold stratified CV per selected model (LogReg pipeline w/ scaling, Random Forest, HistGradientBoosting)."],
    ["Evaluate", "Holdout AUC, PR-AUC, Brier, recall@top-10% — winner = highest holdout AUC."],
    ["Explain", "Permutation importance (global) + median-counterfactual attribution (per user) + exposure lift table (evidence)."],
    ["Segment", "Top-decile risk users get a dominant driver via ordered business rules → action + revenue math."],
    ["Simulate interventions", "Mutate the feature matrix per preset, re-score all holdout users with the winning model, sum the risk deltas."],
    ["Narrate", "Metrics JSON → structured prompt → Claude (or deterministic template) → leadership brief."],
  ];
  $("#pipeline").innerHTML = steps.map(([t, d], i) =>
    `<div class="pipe-step"><div class="pipe-n">STEP ${i + 1}</div><h4>${t}</h4><p>${d}</p></div>`).join("");

  const gloss = [
    ["Churn rate", "share of users with no activity in the following 60 days (simulated label)."],
    ["Recall @ top 10%", "churners inside the riskiest decile ÷ all churners — what a retention team can actually action.", "recall = y[p ≥ q90].sum() / y.sum()"],
    ["Revenue at risk / yr", "each holdout user's churn probability × monthly spend × 12, summed.", "Σ pᵢ · spendᵢ · 12"],
    ["Churn lift (evidence)", "churn rate of users exposed to a pain point ÷ churn rate of everyone else. Observational — confounding is called out where it bites.", "lift = P(churn|exposed) / P(churn|¬exposed)"],
    ["Churners prevented (what-if)", "sum of per-user risk reductions after mutating the feature matrix and re-scoring.", "Σ max(pᵢ − pᵢ′, 0)"],
    ["Revenue protected / yr (what-if)", "the same risk reductions weighted by each user's spend, annualized.", "Σ max(pᵢ − pᵢ′, 0) · spendᵢ · 12"],
    ["Risk contribution (per user)", "user's risk minus their risk with one feature reset to the holdout median — positive means the feature pushes risk up.", "p(x) − p(x | xⱼ → median)"],
    ["Segment revenue protected", "segment monthly spend × assumed save rate × 12 (assumption, pending A/B)."],
  ];
  $("#glossary").innerHTML = gloss.map(([t, how, f]) =>
    `<div class="glossary-item"><div class="g-term">${t}${f ? ` <code>${f}</code>` : ""}</div>
     <div class="g-how">${how}</div></div>`).join("");
}

/* ---------------- narrate ---------------- */
$("#narrate-btn").addEventListener("click", async () => {
  $("#narrate-source").textContent = "Working…";
  const res = await fetch("/api/narrate", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ api_key: $("#api-key").value, run_id: D.run.id }),
  });
  const out = await res.json();
  $("#narrate-source").textContent = "Generated by: " + out.source;
  $("#narrate-output").innerHTML = mdToHtml(out.brief);
});

/* ---------------- boot ---------------- */
$("#in-users").addEventListener("input", () => { $("#out-users").textContent = fmtNum($("#in-users").value); });
$("#run-btn").addEventListener("click", executeRun);
setupTabs();
renderMethodology();
fetch("/api/bootstrap").then((r) => r.json()).then((b) => { render(b); loadRunHistory(); applyRole("ds"); })
  .catch((e) => document.body.insertAdjacentHTML("afterbegin",
    `<div style="background:#e34948;color:#fff;padding:10px 16px">Failed to load data: ${e}</div>`));
