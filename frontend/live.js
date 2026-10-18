// Overview, top of the page: try a prompt, watch monitored requests arrive, and a recorded replay
// of what an alert looks like. Everything is built with el(), so prompts are only ever text.

const FEED_MS = 3000;
const FEED_SIZE = 8;
const FEED_FEATURES = 3;
const TRY_FEATURES = 5;
// Below this many baseline prompts in 1,000, a feature is called rare in normal traffic.
const RARE_PER_1000 = 10;
const labelCache = new Map();
const seenRequests = new Set();

/** How common a feature is in normal traffic, in words, from its rate per 1,000 baseline prompts. */
function rateText(per1000) {
  if (per1000 < RARE_PER_1000) return `rare in normal traffic (under ${RARE_PER_1000} in 1,000 prompts)`;
  return `in ${Math.round(per1000)} of every 1,000 normal prompts`;
}

/** Rows not shown yet, oldest first, so the newest ends up on top when each is prepended. */
function newRows(rows, seen) {
  return rows.filter((row) => !seen.has(row.id)).reverse();
}

/** What kind of traffic a stored request was: the demo trickle, a staged scenario, or other. */
function feedTag(scenario) {
  if (scenario === "demo_normal") return "normal traffic";
  return scenario ? `staged scenario: ${scenario}` : "recorded traffic";
}

/** Labels for these feature ids, from the cache or the API (at most 20 ids per call). */
async function labelsFor(ids) {
  const missing = [...new Set(ids)].filter((id) => !labelCache.has(id));
  for (let i = 0; i < missing.length; i += 20) {
    const chunk = missing.slice(i, i + 20);
    try {
      const labels = await api(`/dashboard/labels?ids=${chunk.join(",")}`);
      for (const id of chunk) labelCache.set(id, labels[id] ?? "label unavailable");
    } catch (_) {
      // Labels need the drift job; without it, show the ids alone.
      for (const id of chunk) labelCache.set(id, "label unavailable");
    }
  }
  return new Map(ids.map((id) => [id, labelCache.get(id)]));
}

function feedItem(row, labels) {
  const features = row.features.slice(0, FEED_FEATURES);
  return el(
    "li",
    { class: "feed-item new" },
    el(
      "p",
      { class: "feed-meta" },
      el("span", { class: "feed-tag" }, feedTag(row.scenario)),
      ` · ${fmtClock(parseTime(row.created_at))}`
    ),
    el("p", { class: "feed-prompt" }, row.prompt),
    el(
      "ul",
      { class: "chips" },
      ...features.map((f) =>
        el(
          "li",
          {},
          el("a", { class: "chip", href: `features.html#feature=${f.feature_id}` }, labels.get(f.feature_id))
        )
      )
    )
  );
}

async function updateFeed() {
  const feed = document.getElementById("feed");
  const status = document.getElementById("feed-status");
  try {
    const rows = newRows(await api(`/features/recent?limit=${FEED_SIZE}`), seenRequests);
    const labels = await labelsFor(rows.flatMap((r) => r.features.slice(0, FEED_FEATURES).map((f) => f.feature_id)));
    for (const row of rows) {
      seenRequests.add(row.id);
      feed.prepend(feedItem(row, labels));
    }
    while (feed.children.length > FEED_SIZE) feed.lastElementChild.remove();
    setStatus(status, feed.children.length ? "" : "No monitored requests yet. They appear here as they arrive.");
  } catch (error) {
    setStatus(status, `Could not load recent requests: ${error.message}`, true);
  }
  setTimeout(updateFeed, FEED_MS);
}

function tryItem(feature, labels, rates) {
  return el(
    "li",
    {},
    el(
      "a",
      { class: "try-row", href: `features.html#feature=${feature.feature_id}` },
      el("span", { class: "try-label" }, labels.get(feature.feature_id), " ", el("span", { class: "id" }, `#${feature.feature_id}`)),
      el("span", { class: "try-rate" }, rateText(rates[feature.feature_id] ?? 0))
    )
  );
}

async function tryPrompt(prompt) {
  const button = document.getElementById("try-submit");
  const status = document.getElementById("try-status");
  const list = document.getElementById("try-results");
  button.disabled = true;
  setStatus(status, "Running GPT-2 on your prompt…");
  try {
    const result = await api("/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ prompt, record: false }),
    });
    const features = result.top_features.slice(0, TRY_FEATURES);
    if (!features.length) {
      setStatus(status, "Feature extraction is switched off on this server.");
      return;
    }
    const ids = features.map((f) => f.feature_id);
    const [labels, rates] = await Promise.all([
      labelsFor(ids),
      api(`/dashboard/normal-rates?ids=${ids.join(",")}`).catch(() => ({})),
    ]);
    list.replaceChildren(...features.map((f) => tryItem(f, labels, rates)));
    document.getElementById("try-more").hidden = false;
    setStatus(status, `The ${features.length} strongest features in your prompt, strongest first:`);
  } catch (error) {
    // 429: the queue of prompts is full, or this address sent too many; the server says which.
    setStatus(status, error.status === 429 ? error.detail : `Could not run the prompt: ${error.message}`, true);
  } finally {
    button.disabled = false;
  }
}

/** The recorded replay's caption; every number comes from recorded_example.json. */
function recordedCaption(example) {
  const first = example.first_above;
  const crossed = (request) => (request === null ? "never crossed 1" : `first crossed 1 at request ${request}`);
  return (
    `One saved replay (seed ${example.seed}) of the ${example.scenario} scenario: from request ` +
    `${example.shift_start}, medical questions replace a growing share of normal ones. Dark line: ` +
    "the SAE detector's score with that shift; light line: the same normal prompts in the same " +
    "order, without it. Scores are divided by the threshold, so 1 is the alarm level. With the " +
    `shift the score ${crossed(first.sae_shift)}, helped by a run of normal prompts that were ` +
    `already close to the line; after that the gap keeps growing. Without the shift it ` +
    `${crossed(first.sae_no_shift)}. In the same replay the sentence-embedding detector ` +
    `${crossed(first.embedding_mean)}.`
  );
}

async function drawRecorded() {
  let example;
  try {
    const response = await fetch("recorded_example.json");
    example = await response.json();
  } catch (_) {
    return;
  }
  document.getElementById("recorded-caption").textContent = recordedCaption(example);
  const xs = example.request_numbers;
  const line = (values, color, label) => ({
    label,
    endLabel: label,
    data: values.map((y, i) => ({ x: xs[i], y })),
    borderColor: color,
    borderWidth: 2,
    pointRadius: 0,
  });
  const options = baseOptions(xs[0], xs[xs.length - 1], "Score ÷ threshold");
  options.scales.x.ticks.callback = (value) => fmtNum(value, 0);
  options.scales.x.title = { display: true, text: "Request number", color: cssVar("--muted") };
  options.plugins.tooltip.callbacks.title = (items) => `Request ${items[0].parsed.x}`;
  options.plugins.tooltip.filter = (item) => !item.dataset.isThreshold;
  new Chart(document.getElementById("recorded"), {
    type: "line",
    data: {
      datasets: [
        line(example.sae_shift, cssVar("--series-1"), "with shift"),
        line(example.sae_no_shift, cssVar("--series-2"), "no shift"),
        {
          label: "Alarm level",
          isThreshold: true,
          data: [
            { x: xs[0], y: 1 },
            { x: xs[xs.length - 1], y: 1 },
          ],
          borderColor: cssVar("--ink-2"),
          borderWidth: 1.5,
          borderDash: [6, 4],
          pointRadius: 0,
          pointHoverRadius: 0,
        },
      ],
    },
    options,
    plugins: [endLabels],
  });
}

document.getElementById("try-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const prompt = document.getElementById("try-prompt").value.trim();
  if (prompt) tryPrompt(prompt);
});
updateFeed();
drawRecorded();
