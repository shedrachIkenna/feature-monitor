// Features page: the latest window's top drifting SAE features, and one feature's detail.

const TABLE_LIMIT = 15;
let rateChart = null;
let selectedId = null;
// Normal rate per feature from the latest table, for the detail chart's dashed line.
const baselineRates = new Map();

applyTerms();

function selectedFromHash() {
  const match = location.hash.match(/^#feature=(\d+)$/);
  return match ? Number(match[1]) : null;
}

function pText(p) {
  return p < 0.001 ? p.toExponential(1) : fmtNum(p, 3);
}

function featureRow(f) {
  const row = el(
    "tr",
    { tabindex: "0", "data-id": f.id, "aria-selected": String(f.id === selectedId) },
    el("td", { class: "label" }, f.label, " ", el("span", { class: "id" }, `#${f.id}`)),
    el("td", { class: "num" }, fmtNum(f.baseline_rate)),
    el("td", { class: "num" }, fmtNum(f.window_rate)),
    el(
      "td",
      {
        class: "num",
        title: `Adjusted p = ${pText(f.adjusted_p)}: the chance of a change at least this large if nothing had really changed, corrected for testing thousands of features at once.`,
      },
      fmtRatio(f.ratio)
    )
  );
  const open = () => select(f.id);
  row.addEventListener("click", open);
  row.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      open();
    }
  });
  return row;
}

function featureTable(caption, features) {
  const head = el(
    "tr",
    {},
    el("th", {}, term("Feature", "feature")),
    el("th", { class: "num" }, term("Normal rate", "baseline")),
    el("th", { class: "num" }, term("Current rate", "rate")),
    el("th", { class: "num" }, term("Ratio", "ratio"))
  );
  const body = features.length
    ? features.map(featureRow)
    : [el("tr", {}, el("td", { colspan: "4", class: "status" }, "None significant."))];
  return el(
    "div",
    {},
    el("h3", {}, caption),
    el("div", { class: "table-wrap" }, el("table", {}, el("thead", {}, head), el("tbody", {}, ...body)))
  );
}

async function loadTable() {
  const status = document.getElementById("table-status");
  const tables = document.getElementById("tables");
  const drifting = await api(`/dashboard/drifting?limit=${TABLE_LIMIT}`);
  if (drifting.last_request_id === null) {
    tables.replaceChildren();
    setStatus(
      status,
      `Waiting for a full window: ${drifting.request_count} of ${drifting.window_size} requests so far.`
    );
    return [];
  }
  setStatus(status, "");
  document.getElementById("window-info").textContent =
    `Rates are per 1,000 requests; this window is requests ${drifting.first_request_id}–${drifting.last_request_id}.`;
  for (const f of [...drifting.rising, ...drifting.falling]) baselineRates.set(f.id, f.baseline_rate);
  tables.replaceChildren(
    featureTable("More common than normal", drifting.rising),
    featureTable("Less common than normal", drifting.falling)
  );
  return drifting.rising;
}

function drawRate(feature, bucketMs) {
  const xMin = parseTime(feature.since);
  const xMax = parseTime(feature.until);
  const color = cssVar("--series-1");
  const points = feature.buckets
    .filter((b) => b.rate !== null)
    .map((b) => ({ x: parseTime(b.start) + bucketMs / 2, y: b.rate, ...b }));
  const datasets = [
    {
      label: "Rate",
      data: points,
      borderColor: color,
      backgroundColor: color,
      borderWidth: 2,
      spanGaps: 3 * bucketMs,
      pointRadius: 0,
      pointHoverRadius: 5,
    },
  ];
  const baseline = baselineRates.get(feature.feature_id);
  if (baseline !== undefined) {
    datasets.push({
      label: "Normal rate",
      endLabel: "normal rate",
      isBaseline: true,
      data: [
        { x: xMin, y: baseline },
        { x: xMax, y: baseline },
      ],
      borderColor: cssVar("--series-2"),
      borderWidth: 1.5,
      borderDash: [6, 4],
      pointRadius: 0,
      pointHoverRadius: 0,
    });
  }
  const options = baseOptions(xMin, xMax, "Per 1,000 requests");
  options.plugins.tooltip.filter = (item) => !item.dataset.isBaseline;
  options.plugins.tooltip.callbacks.label = (item) => {
    const p = item.raw;
    return `${fmtNum(p.rate)} per 1,000 (${p.hits} of ${p.request_count} requests)`;
  };
  options.layout = { padding: { top: 12 } };
  rateChart = drawChart(rateChart, document.getElementById("feature-rate"), {
    type: "line",
    data: { datasets },
    options,
    plugins: [endLabels],
  });
  setStatus(
    document.getElementById("detail-status"),
    points.length ? "" : "No requests in this time range."
  );
}

function drawPrompts(feature) {
  const strong = feature.strong_prompts;
  const count = strong.length ? `${strong.length} ` : "";
  document.getElementById("prompts-sub").textContent =
    `The ${count}newest prompts where this feature was among the ${feature.strong_max_rank} strongest of the prompt's top 20 features.`;
  const items = strong.map((p) =>
    el(
      "li",
      {},
      el("div", {}, p.prompt),
      el(
        "div",
        { class: "meta" },
        `Rank ${p.rank}, activation ${fmtNum(p.activation, 2)}, ${fmtDateTime(parseTime(p.created_at))}`
      )
    )
  );
  if (!items.length) {
    items.push(
      el(
        "li",
        { class: "status" },
        `No stored prompt has this feature among its ${feature.strong_max_rank} strongest. It may fire often but only weakly.`
      )
    );
  }
  document.getElementById("prompts").replaceChildren(...items);
}

async function loadDetail(range) {
  if (selectedId === null) return;
  const feature = await api(
    `/dashboard/feature/${selectedId}?hours=${range.hours}&bucket_minutes=${range.bucketMinutes}`
  );
  const detail = document.getElementById("detail");
  detail.hidden = false;
  document.getElementById("detail-title").textContent =
    `How often feature #${feature.feature_id} appears over time`;
  const { neuronpedia_id: source } = await serverHealth();
  const sub = document.getElementById("detail-sub");
  sub.replaceChildren(
    `"${feature.label}". Solid line: rate per 1,000 requests in each time step; dashed line: normal rate. Steps with few requests are noisy. `,
    el("a", { href: `https://www.neuronpedia.org/${source}/${feature.feature_id}`, target: "_blank", rel: "noopener" }, "Open on Neuronpedia")
  );
  drawRate(feature, feature.bucket_minutes * 60000);
  drawPrompts(feature);
}

function select(id) {
  selectedId = id;
  history.replaceState(null, "", `#feature=${id}`);
  for (const row of document.querySelectorAll("tbody tr[data-id]")) {
    row.setAttribute("aria-selected", String(Number(row.dataset.id) === id));
  }
  loadDetail(currentRange())
    .then(() => document.getElementById("detail").scrollIntoView({ behavior: "smooth", block: "nearest" }))
    .catch(showError);
}

function showError(error) {
  setStatus(document.getElementById("table-status"), `Could not load data: ${error.message}`, true);
}

async function load(range) {
  try {
    const rising = await loadTable();
    // Open the top rising feature when nothing has been picked yet.
    if (selectedId === null && rising.length) {
      selectedId = rising[0].id;
      for (const row of document.querySelectorAll(`tbody tr[data-id="${selectedId}"]`)) {
        row.setAttribute("aria-selected", "true");
      }
    }
    await loadDetail(range);
    document.getElementById("updated").textContent =
      `Updated ${fmtDateTime(Date.now())}; refreshes every ${REFRESH_MS / 1000} s.`;
  } catch (error) {
    showError(error);
  }
}

selectedId = selectedFromHash();
const currentRange = setupRange(load);
setInterval(() => load(currentRange()), REFRESH_MS);
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => load(currentRange()));
