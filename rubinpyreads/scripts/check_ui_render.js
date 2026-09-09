#!/usr/bin/env node
// Laufzeit-Smoke der rubin-UI: transpiliert das Bundle mit dem AUSGELIEFERTEN
// Babel und mountet die komplette App headless (react-test-renderer).
// Fängt die Fehlerklasse, die der Transpile-Check (Ebene 0) NICHT sieht:
// ReferenceErrors durch entfernte Definitionen bei noch vorhandener Nutzung,
// Hook-Reihenfolge-Brüche (TDZ: "Cannot access 'x' before initialization"),
// Render-Crashes. Reale Vorfälle: unknownExcludes nach Segment-Edit entfernt;
// useEffect([done]) vor der done-Deklaration in PRun.
//
// Voraussetzung (einmalig, Versionen == app/frontend/lib):
//   npm i react@18.2.0 react-dom@18.2.0 react-test-renderer@18.2.0
// Nutzung:  node scripts/check_ui_render.js
const path = require("path");
const fs = require("fs");
const ROOT = path.resolve(__dirname, "..");

let TestRenderer, React;
try {
  React = require("react");
  TestRenderer = require("react-test-renderer");
} catch (e) {
  console.log("~ Render-Smoke übersprungen: react/react-test-renderer nicht installiert");
  console.log("  (npm i react@18.2.0 react-dom@18.2.0 react-test-renderer@18.2.0)");
  process.exit(0);
}
if (React.version !== "18.2.0")
  console.log(`~ Hinweis: installiertes React ${React.version} != ausgeliefertes 18.2.0`);

const Babel = require(path.join(ROOT, "app", "frontend", "lib", "babel.min.js"));
const src = fs.readFileSync(path.join(ROOT, "app", "rubin_ui_src.jsx"), "utf8");

// Browser-Umgebung stubben (nur was die App beim Mount berührt)
let captured = null;
const ReactDOM = { createRoot: () => ({ render: (el) => { captured = el; } }) };
global.React = React; global.ReactDOM = ReactDOM;
global.window = global;
global.document = { getElementById: () => ({}), createElement: () => ({}),
                    addEventListener: () => {}, documentElement: {}, body: {} };
global.localStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
global.fetch = () => new Promise(() => {});
global.navigator = { userAgent: "node" };
global.scrollTo = () => {};
global.matchMedia = () => ({ matches: false, addListener: () => {}, removeListener: () => {} });
global.requestAnimationFrame = (cb) => setTimeout(cb, 0);

try {
  const { code } = Babel.transform(src, { presets: ["react"] });
  eval(code);
  if (!captured) { console.error("✗ Render-Smoke: App-Element wurde nicht gemountet"); process.exit(1); }
  TestRenderer.act(() => { TestRenderer.create(captured); });
  console.log("✓ Render-Smoke: App mountet fehlerfrei (headless, React " + React.version + ")");
  process.exit(0);  // App-Timer (Polling) halten node sonst offen
} catch (e) {
  console.error("✗ Render-Smoke: " + (e.message || e));
  console.error((e.stack || "").split("\n").slice(0, 4).join("\n"));
  process.exit(1);
}
