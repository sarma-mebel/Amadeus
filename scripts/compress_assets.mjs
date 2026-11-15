#!/usr/bin/env node
/**
 * Compress the large environment GLBs under static/environment/ with
 * gltf-transform (geometry quantization + Meshopt + texture resizing), so
 * the desktop app's cold-boot loading bar — and every web/mobile load —
 * has less to download and parse. Typical result: ~50-65% smaller.
 *
 * The VRM character is deliberately NOT touched by this script: an earlier
 * attempt showed gltf-transform doesn't understand the VRM-specific
 * extensions (VRMC_vrm, VRMC_springBone, VRMC_materials_mtoon) and silently
 * drops them, which would strip spring-bone physics and toon shading from
 * Kurisu. Environment GLBs are plain glTF with no such extensions, so
 * they're safe to run through the general-purpose pipeline.
 *
 * Usage: node scripts/compress_assets.mjs [--check]
 *   --check   report projected sizes without overwriting anything
 *
 * Always diff a render (see README "Asset compression") before trusting a
 * new environment asset through this script — texture resizing especially
 * can degrade detail depending on the source art.
 */
import { NodeIO } from '@gltf-transform/core'
import { ALL_EXTENSIONS } from '@gltf-transform/extensions'
import { dedup, flatten, join as joinMeshes, meshopt, quantize, textureCompress, weld } from '@gltf-transform/functions'
import { MeshoptDecoder, MeshoptEncoder } from 'meshoptimizer'
import { existsSync, statSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = dirname(fileURLToPath(import.meta.url))
const repoRoot = join(__dirname, '..')
const checkOnly = process.argv.includes('--check')

const ENVIRONMENT_TARGETS = [
  'static/environment/steinsgate_time_machine.glb',
  'static/environment/steinsgate_-_future_gadget_lab.glb',
]

function humanSize(bytes) {
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
}

async function compress(io, relPath) {
  const absPath = join(repoRoot, relPath)
  if (!existsSync(absPath)) {
    console.log(`skip (not present): ${relPath}`)
    return
  }
  const beforeBytes = statSync(absPath).size

  const document = await io.read(absPath)
  await document.transform(
    dedup(),
    weld(),
    // Draw-call reduction: flatten() collapses the (often deeply nested)
    // authoring hierarchy so join() can then merge compatible primitives —
    // same material, same attribute layout — into single meshes. The lab
    // GLB ships as ~250 separate primitives (≈250 draw calls per frame);
    // most share one of only 4 materials, so joining collapses it to a
    // handful. Runs AFTER dedup/weld (so shared geometry is unified first)
    // and BEFORE quantize/meshopt (so the joined geometry is what gets
    // encoded). join() only merges primitives with identical materials, so
    // the transparent BLEND material stays one (joined) mesh of its own —
    // verified visually that its within-material sorting survives for
    // these sets; if a future set's transparency breaks under join, scope
    // it out by name via join({ keepNamed: true }) + naming those nodes.
    flatten(),
    joinMeshes(),
    quantize(),
    meshopt({ encoder: MeshoptEncoder }),
    // targetFormat 'webp' re-encodes the plain 1024x1024 PNG textures
    // (~5.6MB each decoded) as WebP at quality 85 — meaningfully smaller on
    // disk/over-the-wire. NOTE: this is a file-size win only. WebP still
    // decodes to full RGBA in VRAM just like the PNG did (browsers/three.js
    // upload it as an uncompressed texture) — an actual GPU memory
    // reduction needs a real compressed-texture format like KTX2/BasisU,
    // which requires the `toktx` binary (not a JS dependency
    // @gltf-transform/cli can pull in on its own) and was intentionally not
    // pursued here per the "no new heavy deps" constraint.
    textureCompress({ resize: [1024, 1024], targetFormat: 'webp', quality: 85 }),
  )

  if (checkOnly) {
    const bytes = (await io.writeBinary(document)).byteLength
    console.log(`${relPath}: ${humanSize(beforeBytes)} -> ${humanSize(bytes)} (dry run, not written)`)
    return
  }

  await io.write(absPath, document)
  const afterBytes = statSync(absPath).size
  const pct = (100 * (1 - afterBytes / beforeBytes)).toFixed(0)
  console.log(`${relPath}: ${humanSize(beforeBytes)} -> ${humanSize(afterBytes)} (-${pct}%)`)
}

async function main() {
  await MeshoptEncoder.ready
  await MeshoptDecoder.ready
  const io = new NodeIO()
    .registerExtensions(ALL_EXTENSIONS)
    // Both encoder and decoder are needed: the targets already carry
    // EXT_meshopt_compression from a prior run of this same script, so
    // reading them back in (to re-quantize/re-texture-compress) requires
    // the decoder — without it, NodeIO.read() throws immediately on any
    // GLB that's already been through this pipeline once.
    .registerDependencies({ 'meshopt.decoder': MeshoptDecoder, 'meshopt.encoder': MeshoptEncoder })

  for (const target of ENVIRONMENT_TARGETS) {
    await compress(io, target)
  }

  if (!checkOnly) {
    console.log('\nDone. Load the app and compare a screenshot before committing new assets.')
  }
}

main()
