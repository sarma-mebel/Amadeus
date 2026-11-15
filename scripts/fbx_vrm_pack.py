#!/usr/bin/env python3
"""Convert a Bip001-rigged game FBX to VRM 1.0 (avatar, not animation).

Two steps — first flatten the FBX to glb with Facebook's FBX2glTF (the
Blender FBX importer chokes on the in-between morph shapes these game rips
use, and its glTF importer mangles the Biped rest pose; FBX2glTF handles
both), then run this script to inject the VRMC_vrm extension into the glb.
No geometry is touched, so whatever renders correctly as glb renders
correctly as VRM:

    FBX2glTF --binary --input model.fbx --output model
    python scripts/fbx_vrm_pack.py model.glb out.vrm "Display Name" \
        scripts/fbx_vrm_expressions.example.json 100

Args: in.glb out.vrm "Display Name" [expressions.json|""] [root_scale]
- expressions.json: {"aa": [["MeshName"|"*", morphIndex, weight]], ...} —
  "*" = first mesh with morph targets. The example file matches the 9-morph
  face template these game models share (eyebrow01/02 sad/angry brows,
  eye01 blink, eye02 half-lid, eye03 happy-closed, mouth01-04 + extra:
  small-open/smile/rounded-O/wide-open/open-teeth-smile).
- root_scale: 100 for FBX files authored in centimeters (the usual case).

Verified 2026-07-19 on the Rikka/Kurumi/Asuna DeviantArt packs: full
humanoid incl. fingers, 13 expressions, works in-app with idle animations.
Known gaps: no springbones, PBR instead of MToon, no lookAt.
"""
import json
import struct
import sys

BIP = {
    "hips": "Bip001 Pelvis",
    "spine": "Bip001 Spine",
    "chest": "Bip001 Spine1",
    "neck": "Bip001 Neck",
    "head": "Bip001 Head",
    "leftUpperLeg": "Bip001 L Thigh", "rightUpperLeg": "Bip001 R Thigh",
    "leftLowerLeg": "Bip001 L Calf", "rightLowerLeg": "Bip001 R Calf",
    "leftFoot": "Bip001 L Foot", "rightFoot": "Bip001 R Foot",
    "leftToes": "Bip001 L Toe0", "rightToes": "Bip001 R Toe0",
    "leftShoulder": "Bip001 L Clavicle", "rightShoulder": "Bip001 R Clavicle",
    "leftUpperArm": "Bip001 L UpperArm", "rightUpperArm": "Bip001 R UpperArm",
    "leftLowerArm": "Bip001 L Forearm", "rightLowerArm": "Bip001 R Forearm",
    "leftHand": "Bip001 L Hand", "rightHand": "Bip001 R Hand",
}
FINGER_SEGS = {"thumb": ["Metacarpal", "Proximal", "Distal"],
               "other": ["Proximal", "Intermediate", "Distal"]}
FINGER_NAMES = [("Thumb", "Finger0"), ("Index", "Finger1"), ("Middle", "Finger2"),
                ("Ring", "Finger3"), ("Little", "Finger4")]

# Mixamo/Unity-style humanoid naming (Slime-game rips etc.)
STD = {
    "hips": "Hips", "spine": "Spine", "chest": "Spine1", "upperChest": "Spine2",
    "neck": "Neck", "head": "Head",
    "leftUpperLeg": "LeftUpLeg", "rightUpperLeg": "RightUpLeg",
    "leftLowerLeg": "LeftLeg", "rightLowerLeg": "RightLeg",
    "leftFoot": "LeftFoot", "rightFoot": "RightFoot",
    "leftToes": "LeftToeBase", "rightToes": "RightToeBase",
    "leftShoulder": "LeftShoulder", "rightShoulder": "RightShoulder",
    "leftUpperArm": "LeftArm", "rightUpperArm": "RightArm",
    "leftLowerArm": "LeftForeArm", "rightLowerArm": "RightForeArm",
    "leftHand": "LeftHand", "rightHand": "RightHand",
}
STD_FINGERS = ["Thumb", "Index", "Middle", "Ring", "Pinky"]


def main() -> int:
    src, dst, display_name = sys.argv[1], sys.argv[2], sys.argv[3]
    expr_path = sys.argv[4] if len(sys.argv) > 4 else None

    with open(src, "rb") as f:
        magic, version, _length = struct.unpack("<4sII", f.read(12))
        assert magic == b"glTF", "not a glb"
        chunks = []
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            clen, ctype = struct.unpack("<I4s", hdr)
            chunks.append((ctype, f.read(clen)))
    gltf = json.loads(chunks[0][1])

    # Drop geometry bound to FBX2glTF's "Default-Material" placeholder —
    # engine helper planes (gag cheeks, dummies) with no texture that
    # render as opaque white patches.
    default_mats = {i for i, m in enumerate(gltf.get("materials", []))
                    if m.get("name") == "Default-Material"}
    if default_mats:
        for mesh in gltf.get("meshes", []):
            kept = [p for p in mesh.get("primitives", [])
                    if p.get("material") not in default_mats]
            if kept:
                mesh["primitives"] = kept
            # meshes left with only junk primitives keep them (harmless to
            # drop the node instead, but nodes may be skin joints) — hide by
            # collapsing to a degenerate empty primitive list is invalid, so
            # detach the mesh from its nodes below.
        for node in gltf.get("nodes", []):
            if "mesh" in node:
                prims = gltf["meshes"][node["mesh"]].get("primitives", [])
                if prims and all(p.get("material") in default_mats for p in prims):
                    del node["mesh"]

    # Game rips misuse COLOR_0 vertex colors as engine-side masks (often
    # black on the face) — three.js multiplies them into the base color, so
    # strip the attribute references.
    for mesh in gltf.get("meshes", []):
        for prim in mesh.get("primitives", []):
            prim.get("attributes", {}).pop("COLOR_0", None)

    # Face overlays (lashes, iris planes) ship with alpha texels but OPAQUE
    # materials — transparent regions render black. Alpha-mask everything.
    # MASK (not BLEND): discards the invisible overlay planes (blush etc.)
    # while keeping mouth/eye interiors opaque. BLEND breaks depth sorting
    # and composites interior planes on top of the face.
    for mat in gltf.get("materials", []):
        if mat.get("alphaMode", "OPAQUE") == "OPAQUE":
            mat["alphaMode"] = "MASK"
            mat["alphaCutoff"] = 0.4
        # Mirrored nodes flip winding on some face meshes; without
        # double-siding the face gets culled and shows the head interior.
        mat["doubleSided"] = True

    node_by_name = {}
    for i, node in enumerate(gltf.get("nodes", [])):
        node_by_name.setdefault(node.get("name", ""), i)

    def resolve(name):
        return node_by_name.get(name, node_by_name.get(name.replace(" ", "_")))

    # Biped-style mapping
    bip_bones = {}
    for vrm_name, bip_name in BIP.items():
        idx = resolve(bip_name)
        if idx is not None:
            bip_bones[vrm_name] = {"node": idx}
    for side, S in [("left", "L"), ("right", "R")]:
        for vrm_f, bip_f in FINGER_NAMES:
            segs = FINGER_SEGS["thumb"] if vrm_f == "Thumb" else FINGER_SEGS["other"]
            base = f"Bip001 {S} {bip_f}"
            for seg, bip_bone in zip(segs, [base, base + "1", base + "2"]):
                idx = resolve(bip_bone)
                if idx is not None:
                    bip_bones[f"{side}{vrm_f}{seg}"] = {"node": idx}

    # Mixamo/Unity-style mapping
    std_bones = {}
    for vrm_name, std_name in STD.items():
        idx = resolve(std_name)
        if idx is not None:
            std_bones[vrm_name] = {"node": idx}
    for side in ("Left", "Right"):
        for f in STD_FINGERS:
            vrm_f = "Little" if f == "Pinky" else f
            segs = FINGER_SEGS["thumb"] if f == "Thumb" else FINGER_SEGS["other"]
            for k, seg in enumerate(segs, start=1):
                idx = resolve(f"{side}Hand{f}{k}")
                if idx is not None:
                    std_bones[f"{side.lower()}{vrm_f}{seg}"] = {"node": idx}

    human_bones = bip_bones if len(bip_bones) >= len(std_bones) else std_bones

    required = {"hips", "spine", "head", "leftUpperArm", "rightUpperArm",
                "leftLowerArm", "rightLowerArm", "leftHand", "rightHand",
                "leftUpperLeg", "rightUpperLeg", "leftLowerLeg",
                "rightLowerLeg", "leftFoot", "rightFoot"}
    missing = required - set(human_bones)
    if missing:
        print("FATAL missing humanoid bones:", missing)
        return 1

    # Expression binds: map mesh name -> node index (mesh nodes)
    mesh_node_by_name = {}
    for i, node in enumerate(gltf.get("nodes", [])):
        if "mesh" in node:
            mesh = gltf["meshes"][node["mesh"]]
            mesh_node_by_name[mesh.get("name", node.get("name", ""))] = i
            mesh_node_by_name.setdefault(node.get("name", ""), i)

    # "*" in a bind = every node whose mesh has morph targets (faces are
    # often split across primitives/nodes; binding one leaves half-applied
    # expressions, e.g. only one eye blinking)
    morph_nodes = []
    for i, node in enumerate(gltf.get("nodes", [])):
        if "mesh" in node:
            prims = gltf["meshes"][node["mesh"]].get("primitives", [])
            if any("targets" in p for p in prims):
                morph_nodes.append(i)

    expressions = {"preset": {}}
    if expr_path:
        spec = json.load(open(expr_path))
        for preset, binds in spec.items():
            morph_binds = []
            for mesh_name, morph_index, weight in binds:
                if mesh_name == "*":
                    targets = morph_nodes
                else:
                    one = mesh_node_by_name.get(mesh_name)
                    targets = [one] if one is not None else []
                if not targets:
                    print("WARN no mesh node for", mesh_name)
                    continue
                for node_idx in targets:
                    morph_binds.append({"node": node_idx, "index": morph_index,
                                        "weight": weight})
            if morph_binds:
                expressions["preset"][preset] = {"morphTargetBinds": morph_binds,
                                                 "isBinary": False}

    # Optional texture override: FBX2glTF embeds a 1x1 placeholder when the
    # FBX's recorded texture path doesn't resolve — swap in the real PNG.
    tex_override = sys.argv[6] if len(sys.argv) > 6 else None
    if tex_override:
        import base64
        with open(tex_override, "rb") as tf:
            uri = "data:image/png;base64," + base64.b64encode(tf.read()).decode()
        for img in gltf.get("images", []):
            if len(img.get("uri", "")) < 2000:  # placeholder-sized data URI
                img["uri"] = uri
                img["name"] = tex_override.rsplit("/", 1)[-1]

    # FBX cm units: uniformly scale the scene roots up to meters. A shared
    # root scale stays consistent through skinning (joints and mesh nodes
    # all live under it).
    scale = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
    if scale != 1.0:
        for scene in gltf.get("scenes", []):
            for root in scene.get("nodes", []):
                node = gltf["nodes"][root]
                s = node.get("scale", [1.0, 1.0, 1.0])
                node["scale"] = [v * scale for v in s]

    gltf.setdefault("extensionsUsed", [])
    if "VRMC_vrm" not in gltf["extensionsUsed"]:
        gltf["extensionsUsed"].append("VRMC_vrm")
    gltf.setdefault("extensions", {})["VRMC_vrm"] = {
        "specVersion": "1.0",
        "meta": {
            "name": display_name,
            "version": "1.0",
            "authors": ["private fan conversion"],
            "licenseUrl": "https://vrm.dev/licenses/1.0/",
            "avatarPermission": "onlyAuthor",
            "commercialUsage": "personalNonProfit",
            "creditNotation": "unnecessary",
            "modification": "prohibited",
        },
        "humanoid": {"humanBones": human_bones},
        "expressions": expressions,
    }

    payload = json.dumps(gltf, separators=(",", ":")).encode()
    payload += b" " * (-len(payload) % 4)
    out = b""
    for ctype, data in chunks[1:]:
        data += b"\x00" * (-len(data) % 4)
        out += struct.pack("<I4s", len(data), ctype) + data
    header = struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(payload) + len(out))
    with open(dst, "wb") as f:
        f.write(header)
        f.write(struct.pack("<I4s", len(payload), b"JSON") + payload)
        f.write(out)
    print(f"packed {dst}: {len(human_bones)} humanoid bones, "
          f"{len(expressions['preset'])} expressions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
