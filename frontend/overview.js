// Overview page: drift scores with thresholds, request volume, and alerts in the chosen range.

const SERIES_VAR = { sae: "--series-1", embedding: "--series-2" };
const ALERT_TOP = 3;
let scoreChart = null;
let volumeChart = null;

for (const [kind, detector] of Object.entries(DETECTORS)) {
  document.getElementById(`help-${kind}`).textContent = detector.help;
}
applyTerms();

function detectorName(kind) {
  return DETECTORS[kind]?.name ?? kind;
}

function scoreDatasets(timeline, bucketMs, xMin, xMax) {
  const accent = cssVar("--accent");
  const datasets = [];
  for (const kind of timeline.detectors) {
    const color = cssVar(SERIES_VAR[kind] ?? "--series-2");
    const points = timeline.buckets
      .filter((b) => b[kind].score !== null)
      .map((b) => ({ x: parseTime(b.start) + bucketMs / 2, y: b[kind].score, ...b[kind] }));
    if (!points.length) continue;
    // Gaps of a few steps (a late check) are bridged; longer ones (no traffic) break the line.
    datasets.push({
      label: detectorName(kind),
      endLabel: detectorName(kind),
      data: points,
      borderColor: color,
      borderWidth: 2,
      spanGaps: 3 * bucketMs,
      pointRadius: (ctx) => (ctx.raw?.alert ? 4 : 0),
      pointHoverRadius: 6,
      pointBackgroundColor: (ctx) => (ctx.raw?.alert ? accent : color),
      pointBorderColor: (ctx) => (ctx.raw?.alert ? accent : color),
    });
    // The threshold as a dashed step line, held flat to both edges of the chart.
    const steps = points.map((p) => ({ x: p.x, y: p.threshold }));
    steps.unshift({ x: xMin, y: steps[0].y });
    steps.push({ x: xMax, y: steps[steps.length - 1].y });
    datasets.push({
      label: `${detectorName(kind)} threshold`,
      isThreshold: true,
      data: steps,
      borderColor: color,
      borderWidth: 1.5,
      borderDash: [6, 4],
      stepped: "after",
      spanGaps: true,
      pointRadius: 0,
      pointHoverRadius: 0,
    });
  }
  return datasets;
}

function drawScores(timeline, bucketMs, xMin, xMax) {
  const status = document.getElementById("scores-status");
  const datasets = scoreDatasets(timeline, bucketMs, xMin, xMax);
  setStatus(
    status,
    datasets.length
      ? ""
      : "No drift checks in this time range yet. Checks start once 500 requests have arrived."
  );
  const options = baseOptions(xMin, xMax, "Drift score");
  options.scales.y.ticks.callback = (value) => fmtNum(value, 3);
  options.plugins.tooltip.filter = (item) => !item.dataset.isThreshold;
  options.plugins.tooltip.callbacks.label = (item) => {
    const p = item.raw;
    const verdict = p.alert ? " (above threshold: alert)" : "";
    return `${item.dataset.label}: ${fmtNum(p.score, 4)}, threshold ${fmtNum(p.threshold, 4)}${verdict}`;
  };
  options.layout = { padding: { right: 8, top: 12 } };
  scoreChart = drawChart(scoreChart, document.getElementById("scores"), {
    type: "line",
    data: { datasets },
    options,
    plugins: [endLabels],
  });
}

function drawVolume(timeline, bucketMs, xMin, xMax) {
  const data = timeline.buckets.map((b) => ({
    x: parseTime(b.start) + bucketMs / 2,
    y: b.request_count,
  }));
  const options = baseOptions(xMin, xMax, "Requests");
  options.scales.y.ticks.precision = 0;
  options.plugins.tooltip.callbacks.label = (item) => `${fmtNum(item.raw.y, 0)} requests`;
  volumeChart = drawChart(volumeChart, document.getElementById("volume"), {
    type: "bar",
    data: {
      datasets: [
        {
          label: "Requests",
          data,
          backgroundColor: cssVar("--bars"),
          borderRadius: 2,
          barPercentage: 0.9,
          categoryPercentage: 1,
        },
      ],
    },
    options,
  });
}

/** Whether each detector's most recent time step with a check was above its threshold. */
function latestAbove(timeline) {
  const above = {};
  for (const kind of timeline.detectors) {
    const last = timeline.buckets.filter((b) => b[kind].score !== null).pop();
    above[kind] = Boolean(last?.[kind].alert);
  }
  return above;
}

function alertItem(alert, stillAbove) {
  const details = alert.details;
  const isSae = alert.kind === "sae";
  const top = (details.top ?? []).slice(0, ALERT_TOP).map((d) => {
    const arrow = d.ratio > 1 ? "↑" : "↓";
    const examples = d.label ? `e.g. "${d.label}"` : "no prompts in this window";
    const what = isSae ? `${d.label} (feature #${d.id})` : `topic cluster ${d.id}, ${examples}`;
    return el("li", {}, `${arrow} ${fmtRatio(d.ratio)} normal rate: ${what}`);
  });
  return el(
    "li",
    { class: "alert" },
    el(
      "div",
      { class: "head" },
      el("strong", {}, detectorName(alert.kind)),
      el(
        "span",
        {},
        `score ${fmtNum(alert.score, 4)} vs threshold ${fmtNum(details.threshold, 4)}`
      ),
      el("span", { class: "when" }, fmtDateTime(parseTime(alert.created_at))),
      stillAbove ? el("span", { class: "tag" }, "Still above threshold in the latest step") : null
    ),
    top.length
      ? el("div", { class: "sub" }, `Biggest changes, requests ${details.first_request_id}–${details.last_request_id}:`)
      : null,
    top.length ? el("ol", {}, ...top) : null
  );
}

function drawAlerts(alerts, timeline) {
  const since = parseTime(timeline.since);
  const inRange = alerts.filter((a) => parseTime(a.created_at) >= since);
  const above = latestAbove(timeline);
  const tagged = new Set();
  const items = inRange.map((alert) => {
    // Only the newest alert of each detector carries the "still above" tag.
    const stillAbove = above[alert.kind] && !tagged.has(alert.kind);
    tagged.add(alert.kind);
    return alertItem(alert, stillAbove);
  });
  document.getElementById("alerts").replaceChildren(...items);
  setStatus(document.getElementById("alerts-status"), items.length ? "" : "No alerts in this time range.");
}

async function load(range) {
  try {
    const [timeline, alerts] = await Promise.all([
      api(`/dashboard/timeline?hours=${range.hours}&bucket_minutes=${range.bucketMinutes}`),
      api("/alerts?limit=500"),
    ]);
    const bucketMs = timeline.bucket_minutes * 60000;
    const xMin = parseTime(timeline.since);
    const xMax = parseTime(timeline.until);
    drawScores(timeline, bucketMs, xMin, xMax);
    drawVolume(timeline, bucketMs, xMin, xMax);
    drawAlerts(alerts, timeline);
    document.getElementById("updated").textContent =
      `Updated ${fmtDateTime(Date.now())}; refreshes every ${REFRESH_MS / 1000} s.`;
  } catch (error) {
    setStatus(document.getElementById("scores-status"), `Could not load data: ${error.message}`, true);
  }
}

const currentRange = setupRange(load);
setInterval(() => load(currentRange()), REFRESH_MS);
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => load(currentRange()));
