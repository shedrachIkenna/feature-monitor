// Playground page: run one prompt (unrecorded) and show its top features as labelled bars.

applyTerms();

const form = document.getElementById("form");
const promptBox = document.getElementById("prompt");
const button = document.getElementById("submit");
const status = document.getElementById("status");

function barItem(feature, strongest) {
  const share = strongest > 0 ? (feature.activation / strongest) * 100 : 0;
  return el(
    "li",
    {},
    el(
      "a",
      { href: `features.html#feature=${feature.feature_id}`, class: "bar-row" },
      el(
        "span",
        { class: "bar-label" },
        el("span", { "data-label": feature.feature_id }, "loading label…"),
        " ",
        el("span", { class: "id" }, `#${feature.feature_id}`)
      ),
      el("span", { class: "bar-track" }, el("span", { class: "bar", style: `width: ${share.toFixed(1)}%` })),
      el("span", { class: "bar-value" }, fmtNum(feature.activation, 2))
    )
  );
}

/** Fill in labels after the bars are drawn; uncached labels come from Neuronpedia and can be slow. */
async function fillLabels(features) {
  const ids = features.map((f) => f.feature_id).join(",");
  let labels = {};
  try {
    labels = await api(`/dashboard/labels?ids=${ids}`);
  } catch (error) {
    setStatus(status, `Could not load feature labels: ${error.message}`, true);
  }
  for (const node of document.querySelectorAll("[data-label]")) {
    node.textContent = labels[node.dataset.label] ?? "label unavailable";
  }
}

async function run(prompt) {
  button.disabled = true;
  setStatus(status, "Running the model…");
  try {
    const result = await api("/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ prompt, record: false }),
    });
    const features = result.top_features;
    const strongest = Math.max(0, ...features.map((f) => f.activation));
    document.getElementById("bars").replaceChildren(...features.map((f) => barItem(f, strongest)));
    document.getElementById("reply").textContent = result.reply;
    document.getElementById("result").hidden = false;
    setStatus(status, features.length ? "" : "Feature extraction is switched off on this server.");
    if (features.length) await fillLabels(features);
  } catch (error) {
    // 429: the server's queue of prompts is full.
    const message =
      error.status === 429 ? "The model is busy; try again in a moment." : `Could not run the prompt: ${error.message}`;
    setStatus(status, message, true);
  } finally {
    button.disabled = false;
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const prompt = promptBox.value.trim();
  if (prompt) run(prompt);
});
