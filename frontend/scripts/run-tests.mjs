#!/usr/bin/env node
/**
 * Drop-guard test runner.
 *
 * This repo runs its dev workstation on a Windows-mounted filesystem
 * (/mnt/c under WSL2). Under that filesystem, jsdom worker startup is slow
 * enough that a plain `vitest run` occasionally times out spinning up a
 * worker for one or more test files — and when that happens, vitest does
 * NOT fail the run. It silently drops the file from the report and exits
 * 0. A single `npm run test:ci` in CI would look green while a third of
 * the suite quietly never ran (measured live on this tree: a bare `vitest
 * run --reporter=json` executed only 30 of 61 discovered *.test.ts(x)
 * files and still reported `"success": true`).
 *
 * This script closes that hole structurally rather than trying to tune
 * timeouts away: it runs the whole suite once (fast path, workers
 * parallel), diffs vitest's own JSON report against a filesystem
 * discovery of every test file, and individually re-runs (single worker,
 * generous timeouts) anything that either never showed up in the report
 * or showed up as a collection-time failure (worker died before any
 * assertion ran — same drop symptom, just with a stub entry left behind
 * instead of no entry at all). Only after every discovered file has
 * produced a genuine pass/fail does this script decide the run's exit
 * code.
 *
 * Wired as `npm run test:ci` — a sibling CI workflow calls that script
 * name directly, so it must not be renamed without updating that
 * workflow too.
 */

import { spawnSync } from 'node:child_process'
import { mkdtempSync, readFileSync, readdirSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, relative } from 'node:path'
import { fileURLToPath } from 'node:url'

const SCRIPT_DIR = fileURLToPath(new URL('.', import.meta.url))
const FRONTEND_DIR = join(SCRIPT_DIR, '..')

// Every test file the suite is supposed to run, independent of whatever
// vitest itself claims it ran. This is the ground truth the JSON report
// gets diffed against. Implemented as a recursive readdir walk rather than
// fs.globSync('src/**/*.test.{ts,tsx}') because globSync only exists on
// Node 22+ and CI runs Node 20 — the walk must keep producing the exact
// same list (src/-prefixed, forward-slashed, sorted).
function discoverTestFiles() {
  const entries = readdirSync(join(FRONTEND_DIR, 'src'), {
    recursive: true,
    withFileTypes: true,
  })
  const files = entries
    .filter((e) => e.isFile() && /\.test\.(ts|tsx)$/.test(e.name))
    // Dirent.parentPath landed in Node 20.12; older 20.x spells it .path.
    .map((e) => join('src', relative(join(FRONTEND_DIR, 'src'), e.parentPath ?? e.path), e.name))
  return files.map((f) => f.split('\\').join('/')).sort()
}

// Runs `npx vitest run` with the given extra args plus a JSON reporter
// pointed at a fresh temp file, and returns the parsed report (or null if
// vitest never produced parseable JSON at all — e.g. it crashed before
// the reporter could flush, which is itself a drop and handled by the
// caller via an empty executed set).
function runVitest(extraArgs, { label }) {
  const tmpDir = mkdtempSync(join(tmpdir(), 'amadeus-vitest-'))
  const outFile = join(tmpDir, 'report.json')
  console.log(`\n[run-tests] ${label}: npx vitest run ${extraArgs.join(' ')}`)
  const vitestArgs = [
    'vitest',
    'run',
    ...extraArgs,
    '--reporter=default',
    '--reporter=json',
    `--outputFile=${outFile}`,
  ]
  // Windows needs two things here, both validated live on the self-hosted
  // runner 2026-08-05: (1) spawning the npx.cmd shim without `shell: true`
  // throws EINVAL outright since node's CVE-2024-27980 fix, which made
  // every test file look "dropped" in ~25ms because vitest never launched;
  // (2) it has to be the npx shim, not `node node_modules/vitest/vitest.mjs`
  // — the direct entry runs, but jest-dom's expect.extend lands on a
  // different chai instance than the one test files see ("Invalid Chai
  // property: toBeInTheDocument", 21 failures in ChatInterface alone,
  // reproducible either way on the same node binary). With shell: true
  // node joins args unquoted, so quote anything spacey defensively.
  const result = spawnSync(
    process.platform === 'win32' ? 'npx.cmd' : 'npx',
    process.platform === 'win32'
      ? vitestArgs.map((a) => (/\s/.test(a) ? `"${a}"` : a))
      : vitestArgs,
    {
      cwd: FRONTEND_DIR,
      stdio: 'inherit',
      shell: process.platform === 'win32',
      // Keep going even if the process errors — the JSON diff, not
      // vitest's own exit code, is what actually decides pass/fail here,
      // since a dropped file exits 0 while a genuinely failed one exits
      // 1; both need the same downstream handling (parse + diff).
    }
  )
  let report = null
  try {
    report = JSON.parse(readFileSync(outFile, 'utf8'))
  } catch {
    report = null
  } finally {
    rmSync(tmpDir, { recursive: true, force: true })
  }
  return { report, exitCode: result.status }
}

// Classifies a single vitest JSON testResults[] entry.
//   'passed'          -> every assertion in the file passed
//   'failed-real'      -> the file ran and at least one assertion failed
//   'failed-to-start'  -> the worker died before any assertion ran
//                         (assertionResults is empty; this is the same
//                         symptom as a full drop, just with a stub left
//                         in the report instead of no entry at all)
function classify(entry) {
  if (entry.status !== 'failed') return 'passed'
  if (!entry.assertionResults || entry.assertionResults.length === 0) {
    return 'failed-to-start'
  }
  const anyFailed = entry.assertionResults.some((a) => a.status === 'failed')
  return anyFailed ? 'failed-real' : 'passed'
}

// Absolute paths in the report -> repo-relative, forward-slashed, so they
// compare cleanly against discoverTestFiles()'s output regardless of the
// OS path separator.
function toRelative(absPath) {
  return relative(FRONTEND_DIR, absPath).split('\\').join('/')
}

function main() {
  const discovered = discoverTestFiles()
  console.log(`[run-tests] discovered ${discovered.length} test files under src/`)

  const { report: firstReport } = runVitest([], { label: 'full suite' })

  // outcomes: relative path -> 'passed' | 'failed-real' | 'failed-to-start' | 'missing'
  const outcomes = new Map(discovered.map((f) => [f, 'missing']))
  if (firstReport) {
    for (const entry of firstReport.testResults ?? []) {
      const rel = toRelative(entry.name)
      if (outcomes.has(rel)) {
        outcomes.set(rel, classify(entry))
      }
      // Entries for files outside the discovered set (shouldn't happen,
      // since vitest is only ever invoked with our own glob) are ignored
      // rather than crashing the script.
    }
  }

  const needsRetry = discovered.filter(
    (f) => outcomes.get(f) === 'missing' || outcomes.get(f) === 'failed-to-start'
  )

  if (needsRetry.length > 0) {
    console.log(
      `\n[run-tests] ${needsRetry.length} file(s) did not produce a real result in the full run ` +
        '(dropped entirely, or the worker died before any assertion ran). ' +
        'Re-running each individually, single-worker, generous timeouts:'
    )
    for (const f of needsRetry) {
      const reason = outcomes.get(f)
      console.log(`  - ${f} (${reason})`)
    }

    for (const f of needsRetry) {
      const { report } = runVitest(
        [
          f,
          '--pool=threads',
          '--maxWorkers=1',
          '--testTimeout=120000',
          '--hookTimeout=120000',
        ],
        { label: `retry: ${f}` }
      )
      const entry = report?.testResults?.find((e) => toRelative(e.name) === f)
      outcomes.set(f, entry ? classify(entry) : 'missing')
    }
  }

  // ---- Summary ----
  console.log('\n[run-tests] ===== per-file summary =====')
  let anyBad = false
  for (const f of discovered) {
    const outcome = outcomes.get(f)
    const badge =
      outcome === 'passed'
        ? 'PASS'
        : outcome === 'failed-real'
          ? 'FAIL'
          : outcome === 'failed-to-start'
            ? 'FAIL (never started)'
            : 'FAIL (missing from every run)'
    if (outcome !== 'passed') anyBad = true
    console.log(`  [${badge}] ${f}`)
  }

  const retried = needsRetry.length
  const recovered = needsRetry.filter((f) => outcomes.get(f) === 'passed').length
  console.log(
    `\n[run-tests] ${discovered.length} discovered, ${retried} needed an individual retry, ` +
      `${recovered} recovered on retry.`
  )

  if (anyBad) {
    console.log('[run-tests] RESULT: FAIL — see per-file summary above.')
    process.exit(1)
  }
  console.log('[run-tests] RESULT: PASS — every discovered file ran and passed.')
  process.exit(0)
}

main()
