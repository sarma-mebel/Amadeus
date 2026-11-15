"""Convert Quaternius UAL (glTF, Rigify DEF- rig) animations to VRMA.

VRMA = GLB with a VRMC_vrm_animation extension mapping VRM humanoid bone
names to skeleton node indices. We keep only the skeleton subtree and one
animation per output file, repacking the referenced accessor bytes into a
fresh buffer. Godot-exported `scale` tracks (rejected by
three-vrm-animation's parser) and non-hips `translation` tracks are
stripped per the VRMA spec.

Source pack (CC0): https://quaternius.com/packs/universalanimationlibrary.html
glTF mirror used: https://github.com/J-Ponzo/gltf-universal-animation-library

Usage:
    python scripts/ual_gltf_to_vrma.py <library.gltf> <library.bin> \
        <AnimationName> <output.vrma> [<AnimationName> <output.vrma> ...]

See static/animations/SOURCES.md for the clips already vendored this way.
"""
import json
import struct
import sys

SRC_GLTF = "ual.gltf"
SRC_BIN = "ual.bin"

# UAL2 uses an UE-style skeleton instead of UAL1's Rigify DEF- rig; the
# right map is picked automatically by which hips bone exists.
UE_BONE_MAP = {
    "pelvis": "hips",
    "spine_01": "spine",
    "spine_02": "chest",
    "spine_03": "upperChest",
    "neck_01": "neck",
    "Head": "head",
    "clavicle_l": "leftShoulder",
    "upperarm_l": "leftUpperArm",
    "lowerarm_l": "leftLowerArm",
    "hand_l": "leftHand",
    "clavicle_r": "rightShoulder",
    "upperarm_r": "rightUpperArm",
    "lowerarm_r": "rightLowerArm",
    "hand_r": "rightHand",
    "thigh_l": "leftUpperLeg",
    "calf_l": "leftLowerLeg",
    "foot_l": "leftFoot",
    "ball_l": "leftToes",
    "thigh_r": "rightUpperLeg",
    "calf_r": "rightLowerLeg",
    "foot_r": "rightFoot",
    "ball_r": "rightToes",
    "thumb_01_l": "leftThumbMetacarpal",
    "thumb_02_l": "leftThumbProximal",
    "thumb_03_l": "leftThumbDistal",
    "index_01_l": "leftIndexProximal",
    "index_02_l": "leftIndexIntermediate",
    "index_03_l": "leftIndexDistal",
    "middle_01_l": "leftMiddleProximal",
    "middle_02_l": "leftMiddleIntermediate",
    "middle_03_l": "leftMiddleDistal",
    "ring_01_l": "leftRingProximal",
    "ring_02_l": "leftRingIntermediate",
    "ring_03_l": "leftRingDistal",
    "pinky_01_l": "leftLittleProximal",
    "pinky_02_l": "leftLittleIntermediate",
    "pinky_03_l": "leftLittleDistal",
    "thumb_01_r": "rightThumbMetacarpal",
    "thumb_02_r": "rightThumbProximal",
    "thumb_03_r": "rightThumbDistal",
    "index_01_r": "rightIndexProximal",
    "index_02_r": "rightIndexIntermediate",
    "index_03_r": "rightIndexDistal",
    "middle_01_r": "rightMiddleProximal",
    "middle_02_r": "rightMiddleIntermediate",
    "middle_03_r": "rightMiddleDistal",
    "ring_01_r": "rightRingProximal",
    "ring_02_r": "rightRingIntermediate",
    "ring_03_r": "rightRingDistal",
    "pinky_01_r": "rightLittleProximal",
    "pinky_02_r": "rightLittleIntermediate",
    "pinky_03_r": "rightLittleDistal",
}

BONE_MAP = {
    "DEF-hips": "hips",
    "DEF-spine.001": "spine",
    "DEF-spine.002": "chest",
    "DEF-spine.003": "upperChest",
    "DEF-neck": "neck",
    "DEF-head": "head",
    "DEF-shoulder.L": "leftShoulder",
    "DEF-upper_arm.L": "leftUpperArm",
    "DEF-forearm.L": "leftLowerArm",
    "DEF-hand.L": "leftHand",
    "DEF-shoulder.R": "rightShoulder",
    "DEF-upper_arm.R": "rightUpperArm",
    "DEF-forearm.R": "rightLowerArm",
    "DEF-hand.R": "rightHand",
    "DEF-thigh.L": "leftUpperLeg",
    "DEF-shin.L": "leftLowerLeg",
    "DEF-foot.L": "leftFoot",
    "DEF-toe.L": "leftToes",
    "DEF-thigh.R": "rightUpperLeg",
    "DEF-shin.R": "rightLowerLeg",
    "DEF-foot.R": "rightFoot",
    "DEF-toe.R": "rightToes",
    "DEF-thumb.01.L": "leftThumbMetacarpal",
    "DEF-thumb.02.L": "leftThumbProximal",
    "DEF-thumb.03.L": "leftThumbDistal",
    "DEF-f_index.01.L": "leftIndexProximal",
    "DEF-f_index.02.L": "leftIndexIntermediate",
    "DEF-f_index.03.L": "leftIndexDistal",
    "DEF-f_middle.01.L": "leftMiddleProximal",
    "DEF-f_middle.02.L": "leftMiddleIntermediate",
    "DEF-f_middle.03.L": "leftMiddleDistal",
    "DEF-f_ring.01.L": "leftRingProximal",
    "DEF-f_ring.02.L": "leftRingIntermediate",
    "DEF-f_ring.03.L": "leftRingDistal",
    "DEF-f_pinky.01.L": "leftLittleProximal",
    "DEF-f_pinky.02.L": "leftLittleIntermediate",
    "DEF-f_pinky.03.L": "leftLittleDistal",
    "DEF-thumb.01.R": "rightThumbMetacarpal",
    "DEF-thumb.02.R": "rightThumbProximal",
    "DEF-thumb.03.R": "rightThumbDistal",
    "DEF-f_index.01.R": "rightIndexProximal",
    "DEF-f_index.02.R": "rightIndexIntermediate",
    "DEF-f_index.03.R": "rightIndexDistal",
    "DEF-f_middle.01.R": "rightMiddleProximal",
    "DEF-f_middle.02.R": "rightMiddleIntermediate",
    "DEF-f_middle.03.R": "rightMiddleDistal",
    "DEF-f_ring.01.R": "rightRingProximal",
    "DEF-f_ring.02.R": "rightRingIntermediate",
    "DEF-f_ring.03.R": "rightRingDistal",
    "DEF-f_pinky.01.R": "rightLittleProximal",
    "DEF-f_pinky.02.R": "rightLittleIntermediate",
    "DEF-f_pinky.03.R": "rightLittleDistal",
}

COMPONENT_SIZE = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
TYPE_COUNT = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def accessor_bytes(gltf, bin_data, index):
    acc = gltf["accessors"][index]
    view = gltf["bufferViews"][acc["bufferView"]]
    stride = view.get("byteStride")
    elem = COMPONENT_SIZE[acc["componentType"]] * TYPE_COUNT[acc["type"]]
    start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    if stride and stride != elem:
        out = bytearray()
        for i in range(acc["count"]):
            off = start + i * stride
            out += bin_data[off : off + elem]
        return bytes(out)
    return bin_data[start : start + elem * acc["count"]]


def convert(anim_name, out_path):
    gltf = json.load(open(SRC_GLTF))
    bin_data = open(SRC_BIN, "rb").read()
    nodes = gltf["nodes"]

    anim = next(a for a in gltf["animations"] if a["name"] == anim_name)

    # Skeleton subtree: hips and all descendants, plus ancestor chain.
    name_to_index = {n.get("name"): i for i, n in enumerate(nodes)}
    bone_map = BONE_MAP if "DEF-hips" in name_to_index else UE_BONE_MAP
    hips = name_to_index["DEF-hips" if "DEF-hips" in name_to_index else "pelvis"]
    keep = set()

    def walk(i):
        keep.add(i)
        for c in nodes[i].get("children", []):
            walk(c)

    walk(hips)
    # ancestors (root, Rig) so any world transform is preserved
    changed = True
    while changed:
        changed = False
        for i, n in enumerate(nodes):
            if i not in keep and any(c in keep for c in n.get("children", [])):
                keep.add(i)
                changed = True

    old_to_new = {old: new for new, old in enumerate(sorted(keep))}
    new_nodes = []
    for old in sorted(keep):
        n = dict(nodes[old])
        n.pop("mesh", None)
        n.pop("skin", None)
        if "children" in n:
            kids = [old_to_new[c] for c in n["children"] if c in keep]
            if kids:
                n["children"] = kids
            else:
                n.pop("children")
        new_nodes.append(n)

    root_new = [old_to_new[i] for i in keep if not any(i in nodes[p].get("children", []) for p in keep)]

    # Animation: keep channels that target kept nodes; repack accessors.
    new_buffer = bytearray()
    new_views = []
    new_accessors = []
    accessor_cache = {}

    def repack(index):
        if index in accessor_cache:
            return accessor_cache[index]
        data = accessor_bytes(gltf, bin_data, index)
        offset = len(new_buffer)
        new_buffer.extend(data)
        while len(new_buffer) % 4:
            new_buffer.append(0)
        view_index = len(new_views)
        new_views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(data)})
        src = gltf["accessors"][index]
        acc = {
            "bufferView": view_index,
            "componentType": src["componentType"],
            "count": src["count"],
            "type": src["type"],
        }
        for key in ("min", "max", "normalized"):
            if key in src:
                acc[key] = src[key]
        new_index = len(new_accessors)
        new_accessors.append(acc)
        accessor_cache[index] = new_index
        return new_index

    new_samplers = []
    new_channels = []
    for channel in anim["channels"]:
        target_node = channel["target"].get("node")
        if target_node not in keep:
            continue
        # VRMA allows rotation tracks on humanoid bones and translation only
        # on hips; Godot bakes scale (and translation) tracks on every bone
        # and three-vrm-animation's parser throws on "scale".
        path = channel["target"]["path"]
        if path == "scale":
            continue
        if path == "translation" and target_node != hips:
            continue
        sampler = anim["samplers"][channel["sampler"]]
        new_sampler_index = len(new_samplers)
        new_samplers.append(
            {
                "input": repack(sampler["input"]),
                "output": repack(sampler["output"]),
                "interpolation": sampler.get("interpolation", "LINEAR"),
            }
        )
        new_channels.append(
            {
                "sampler": new_sampler_index,
                "target": {"node": old_to_new[target_node], "path": channel["target"]["path"]},
            }
        )

    human_bones = {}
    for old in keep:
        vrm_name = bone_map.get(nodes[old].get("name"))
        if vrm_name:
            human_bones[vrm_name] = {"node": old_to_new[old]}

    doc = {
        "asset": {"version": "2.0", "generator": "amadeus ual2vrma"},
        "extensionsUsed": ["VRMC_vrm_animation"],
        "extensions": {
            "VRMC_vrm_animation": {
                "specVersion": "1.0",
                "humanoid": {"humanBones": human_bones},
            }
        },
        "scene": 0,
        "scenes": [{"nodes": root_new}],
        "nodes": new_nodes,
        "animations": [
            {"name": anim_name, "samplers": new_samplers, "channels": new_channels}
        ],
        "buffers": [{"byteLength": len(new_buffer)}],
        "bufferViews": new_views,
        "accessors": new_accessors,
    }

    json_bytes = json.dumps(doc, separators=(",", ":")).encode()
    while len(json_bytes) % 4:
        json_bytes += b" "
    bin_bytes = bytes(new_buffer)
    while len(bin_bytes) % 4:
        bin_bytes += b"\0"
    glb = (
        struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(json_bytes) + 8 + len(bin_bytes))
        + struct.pack("<II", len(json_bytes), 0x4E4F534A)
        + json_bytes
        + struct.pack("<II", len(bin_bytes), 0x004E4942)
        + bin_bytes
    )
    open(out_path, "wb").write(glb)
    print(f"{out_path}: {len(glb)} bytes, {len(new_channels)} channels, {len(human_bones)} human bones")


if __name__ == "__main__":
    if len(sys.argv) < 5 or len(sys.argv) % 2 == 0:
        print(__doc__)
        sys.exit(1)
    SRC_GLTF, SRC_BIN = sys.argv[1], sys.argv[2]
    pairs = sys.argv[3:]
    for name, out in zip(pairs[0::2], pairs[1::2]):
        convert(name, out)
