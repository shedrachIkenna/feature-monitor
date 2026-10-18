"""Run the dashboard's real scripts in Node with stand-ins for the browser, then evaluate in their scope."""

import json
import shutil
import subprocess

import pytest

from backend.app import FRONTEND_DIR

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")

# Every DOM lookup returns a do-nothing element, fetch always fails and timers never fire, so the
# scripts' top-level code runs without a page or a server.
RUNNER = """
const vm = require("vm");
const fs = require("fs");
const fake = new Proxy(function () {}, {
  get: (_, key) => (key === "children" ? [] : key === Symbol.toPrimitive ? () => "" : fake),
  apply: () => fake,
});
const context = {
  fetch: () => Promise.reject(new Error("no server")),
  setTimeout: () => 0,
  document: fake,
  getComputedStyle: () => fake,
};
vm.createContext(context);
const [expression, ...scripts] = process.argv.slice(1);
for (const script of scripts) vm.runInContext(fs.readFileSync(script, "utf8"), context);
console.log(JSON.stringify(vm.runInContext(expression, context)));
"""


def evaluate(expression: str, *scripts: str):
    """Load the named frontend scripts in order, then return `expression`'s value as JSON."""
    paths = [str(FRONTEND_DIR / s) for s in scripts]
    result = subprocess.run(
        ["node", "-e", RUNNER, expression, *paths], capture_output=True, text=True, check=True
    )
    return json.loads(result.stdout)
