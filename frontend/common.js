// Shared helpers for the dashboard pages: API calls, the time-range picker, formatting,
// chart styling and safe DOM building (prompts are user text, so never use innerHTML for them).

const REFRESH_MS = 30000;
// The banner's warm-up and scenario counts are checked this often while they move.
const DEMO_LIVE_MS = 2000;
const Y_AXIS_WIDTH = 64;

// Each range picks a bucket size the API supports: 1-minute steps for short scenario runs.
const RANGES = {
  "1h": { hours: 1, bucketMinutes: 1, label: "Last hour (1-minute steps)" },
  "6h": { hours: 6, bucketMinutes: 5, label: "Last 6 hours (5-minute steps)" },
  "24h": { hours: 24, bucketMinutes: 5, label: "Last 24 hours (5-minute steps)" },
};
const DEFAULT_RANGE = "1h";
const RANGE_KEY = "feature-monitor-range";

const DETECTORS = {
  sae: {
    name: "SAE features",
    help: "Drift in which internal model features (from the sparse autoencoder) fire on prompts.",
  },
  embedding_mean: {
    name: "Sentence embeddings",
    help: "The simpler baseline: drift in the average meaning of the prompts (distance between the window's mean sentence embedding and normal traffic's).",
  },
  // Saved runs from before 2026-10-16 scored embeddings by topic cluster instead.
  embedding: {
    name: "Sentence embeddings (clusters, old)",
    help: "Drift in which topic cluster each prompt's sentence embedding falls in.",
  },
};

const TERMS = {
  drift:
    "Drift score: how different the latest 500 requests are from normal traffic (Jensen-Shannon divergence; 0 = identical).",
  threshold:
    "Threshold: the 99th percentile of scores on normal traffic, set per detector. A check scoring above it raises an alert.",
  rate: "Rate: how many requests out of every 1,000 had this feature among their top 20.",
  baseline: "Normal rate: the feature's rate in the 3,000 normal prompts the detector was built from.",
  ratio: "Ratio: current rate divided by normal rate. 2.0× means twice as common as normal.",
  feature:
    "Feature: one direction in the model's internal activations that the sparse autoencoder learned; its label comes from Neuronpedia.",
  activation:
    "Activation: how strongly the feature fired on the prompt. Scales differ between features, so compare bars within one prompt, not across features' typical sizes.",
  window: "Window: the latest 500 requests, which every drift check compares against normal traffic.",
};

/** Fetch JSON from the API; throws with the server's message on an error status. */
async function api(path, init) {
  const response = await fetch(path, init);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      // Validation errors (422) carry a list of problems instead of one message.
      detail = Array.isArray(body.detail) ? body.detail.map((d) => d.msg).join("; ") : body.detail || detail;
    } catch (_) {
      // Not JSON; keep the status text.
    }
    const error = new Error(`${path}: ${response.status} ${detail}`);
    error.status = response.status;
    error.detail = detail;
    throw error;
  }
  return response.json();
}

function loadRange() {
  try {
    const key = localStorage.getItem(RANGE_KEY);
    if (key in RANGES) return key;
  } catch (_) {
    // Storage blocked; use the default.
  }
  return DEFAULT_RANGE;
}

function saveRange(key) {
  try {
    localStorage.setItem(RANGE_KEY, key);
  } catch (_) {
    // Storage blocked; the choice lasts until reload.
  }
}

/** Fill the header's range picker and call onChange(range) now and whenever it changes. */
function setupRange(onChange) {
  const select = document.getElementById("range");
  for (const [key, range] of Object.entries(RANGES)) {
    select.append(el("option", { value: key }, range.label));
  }
  select.value = loadRange();
  select.addEventListener("change", () => {
    saveRange(select.value);
    onChange(RANGES[select.value]);
  });
  onChange(RANGES[select.value]);
  return () => RANGES[select.value];
}

/** API timestamps ("2026-10-15T23:17:09.554778Z") to milliseconds since the epoch. */
function parseTime(iso) {
  return Date.parse(iso.replace(/(\.\d{3})\d*Z$/, "$1Z"));
}

function fmtClock(ms) {
  return new Date(ms).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function fmtDateTime(ms) {
  return new Date(ms).toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function fmtNum(x, digits = 1) {
  if (x === null || x === undefined) return "–";
  return x.toLocaleString([], { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

function fmtRatio(x) {
  return `${x >= 10 ? fmtNum(x, 0) : fmtNum(x, x >= 1 ? 1 : 2)}×`;
}

/** Build an element; children are strings (as text, never HTML) or nodes. */
function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value !== null && value !== undefined) node.setAttribute(key, value);
  }
  node.append(...children.filter((c) => c !== null && c !== undefined));
  return node;
}

/** A term with its one-line explanation as hover text. */
function term(text, key) {
  return el("span", { class: "term", title: TERMS[key], tabindex: "0" }, text);
}

/** Give every [data-term] element in the static HTML its hover explanation. */
function applyTerms() {
  for (const node of document.querySelectorAll("[data-term]")) {
    node.classList.add("term");
    node.title = TERMS[node.dataset.term];
    node.tabIndex = 0;
  }
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function setStatus(node, message, isError = false) {
  node.replaceChildren(message ? el("p", { class: isError ? "status error" : "status" }, message) : "");
}

/** Chart.js options shared by every chart: a time x-axis in ms, recessive grid, no animation. */
function baseOptions(xMin, xMax, yTitle) {
  const muted = cssVar("--muted");
  const grid = cssVar("--grid");
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    interaction: { mode: "nearest", axis: "x", intersect: false },
    plugins: {
      legend: { display: false },
      tooltip: { callbacks: { title: (items) => fmtClock(items[0].parsed.x) } },
    },
    scales: {
      x: {
        type: "linear",
        min: xMin,
        max: xMax,
        grid: { display: false },
        border: { color: cssVar("--axis") },
        ticks: { color: muted, maxTicksLimit: 7, callback: (value) => fmtClock(value) },
      },
      y: {
        // A fixed axis width keeps stacked charts' time axes lined up.
        afterFit: (axis) => {
          axis.width = Y_AXIS_WIDTH;
        },
        beginAtZero: true,
        grid: { color: grid },
        border: { display: false },
        ticks: { color: muted, maxTicksLimit: 5 },
        title: { display: Boolean(yTitle), text: yTitle, color: muted },
      },
    },
  };
}

/** Chart.js plugin: write each dataset's `endLabel` beside its last point (direct labelling). */
const endLabels = {
  id: "endLabels",
  afterDatasetsDraw(chart) {
    const { ctx, chartArea } = chart;
    ctx.save();
    ctx.font = "12px system-ui, -apple-system, sans-serif";
    ctx.fillStyle = cssVar("--ink-2");
    ctx.textBaseline = "middle";
    chart.data.datasets.forEach((dataset, i) => {
      const points = chart.getDatasetMeta(i).data;
      if (!dataset.endLabel || !points.length) return;
      const last = points[points.length - 1];
      const width = ctx.measureText(dataset.endLabel).width;
      const x = Math.min(last.x + 6, chartArea.right - width);
      ctx.fillText(dataset.endLabel, x, last.y - 10);
    });
    ctx.restore();
  },
};

/** Replace a chart's contents in place, or create it on first use. */
function drawChart(existing, canvas, config) {
  if (existing) existing.destroy();
  return new Chart(canvas, config);
}

/** How long to wait before checking /demo/status again: often while counts are moving. */
function noticeDelay(demo) {
  return demo && (demo.warmup || demo.scenario) ? DEMO_LIVE_MS : REFRESH_MS;
}

/** The banner under the header: one line for storage that resets, one for the demo traffic. */
function noticeBox() {
  let box = document.getElementById("server-notices");
  if (!box) {
    const line = (id) =>
      el(
        "p",
        { id, class: "notice", role: "note", hidden: "" },
        el("span", { class: "pulse", "aria-hidden": "true" }),
        el("span", { class: "text" }),
      );
    box = el("div", { id: "server-notices" }, line("notice-storage"), line("notice-demo"));
    document.querySelector("header").after(box);
  }
  return box;
}

/** Show `text` on one banner line (hide it when empty); `live` adds the pulsing dot. */
function setNotice(id, text, live = false) {
  const line = document.getElementById(id);
  line.hidden = !text;
  line.classList.toggle("live", live);
  line.querySelector(".text").textContent = text;
}

/** The demo traffic line: warm-up progress, a running scenario, or the steady trickle. */
function demoNotice(demo) {
  if (demo.scenario) {
    return (
      `Demo scenario running: ${demo.scenario.name}, request ${demo.scenario.sent} / ` +
      `${demo.scenario.total}. This drift is staged on purpose to show the detectors at work.`
    );
  }
  if (demo.warmup) {
    return (
      `Warming up: ${demo.warmup.stored} / ${demo.warmup.target} requests stored. Drift checks ` +
      "start once the window is full; until then the charts stay empty."
    );
  }
  return demo.enabled
    ? "Live traffic here is a steady trickle of normal support questions sent by the server itself."
    : "";
}

// /health's storage answer cannot change while the server runs, so it is fetched once.
let storageResets = null;

/** Update the banner, then check again: every 2 s while a count is moving, else every 30 s. */
async function showServerNotices() {
  let demo = null;
  try {
    if (storageResets === null) storageResets = (await api("/health")).storage_resets;
    demo = await api("/demo/status");
    noticeBox();
    setNotice(
      "notice-storage",
      storageResets
        ? "Demo data resets on restart: this server's database is wiped whenever the Space " +
            "restarts, redeploys or wakes from sleep, so history starts empty after that."
        : "",
    );
    setNotice("notice-demo", demoNotice(demo), Boolean(demo.warmup || demo.scenario));
  } catch (_) {
    // Server unreachable for now; try again at the slow pace.
  }
  setTimeout(showServerNotices, noticeDelay(demo));
}

showServerNotices();
