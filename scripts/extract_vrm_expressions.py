#!/usr/bin/env python3
"""
Extract expression/blend shape data from Kurisu.vrm file.
VRM files are glTF with VRM extensions containing expression presets.
"""

import json
from pathlib import Path
from pygltflib import GLTF2

def extract_vrm_expressions(vrm_path):
    """Extract all expression data from VRM file."""
    print("=" * 70)
    print("VRM Expression Extractor for Kurisu")
    print("=" * 70)
    print()

    print(f"Loading VRM file: {vrm_path}")

    # VRM files are binary glTF (GLB format)
    gltf = GLTF2().load_binary(vrm_path)
    print("✅ VRM file loaded successfully (binary GLB format)")
    print()

    # VRM data is stored in extensions
    if not gltf.extensions:
        print("❌ No extensions found in VRM file")
        return

    print("Available extensions:")
    for ext_name in gltf.extensions:
        print(f"  - {ext_name}")
    print()

    # VRM 0.x uses "VRM" extension, VRM 1.x uses "VRMC_vrm"
    vrm_data = None
    vrm_version = None

    if "VRM" in gltf.extensions:
        vrm_data = gltf.extensions["VRM"]
        vrm_version = "0.x"
        print(f"✅ Found VRM {vrm_version} data")
    elif "VRMC_vrm" in gltf.extensions:
        vrm_data = gltf.extensions["VRMC_vrm"]
        vrm_version = "1.x"
        print(f"✅ Found VRM {vrm_version} data")
    else:
        print("❌ No VRM extension data found")
        return

    print()
    print("=" * 70)
    print("Expression Data")
    print("=" * 70)
    print()

    expressions = {}

    # Extract expressions based on VRM version
    if vrm_version == "0.x":
        # VRM 0.x stores expressions in blendShapeMaster.blendShapeGroups
        if "blendShapeMaster" in vrm_data:
            blend_shape_groups = vrm_data["blendShapeMaster"].get("blendShapeGroups", [])

            print(f"Found {len(blend_shape_groups)} blend shape groups (expressions)")
            print()

            for group in blend_shape_groups:
                name = group.get("name", "unknown")
                preset_name = group.get("presetName", "unknown")
                binds = group.get("binds", [])

                expressions[name] = {
                    "name": name,
                    "preset": preset_name,
                    "blend_count": len(binds),
                    "is_binary": group.get("isBinary", False)
                }

                print(f"Expression: {name}")
                print(f"  Preset: {preset_name}")
                print(f"  Blend shapes: {len(binds)}")
                print(f"  Binary: {group.get('isBinary', False)}")
                print()

    elif vrm_version == "1.x":
        # VRM 1.x stores expressions in expressions
        if "expressions" in vrm_data:
            expr_data = vrm_data["expressions"]

            # VRM 1.x has preset and custom expressions
            preset_expressions = expr_data.get("preset", {})
            custom_expressions = expr_data.get("custom", [])

            print(f"Found {len(preset_expressions)} preset expressions")
            print(f"Found {len(custom_expressions)} custom expressions")
            print()

            # Preset expressions
            for expr_name, expr_info in preset_expressions.items():
                expressions[expr_name] = {
                    "name": expr_name,
                    "preset": expr_name,
                    "type": "preset",
                    "is_binary": expr_info.get("isBinary", False)
                }

                print(f"Preset Expression: {expr_name}")
                print(f"  Binary: {expr_info.get('isBinary', False)}")
                print()

            # Custom expressions
            for i, expr_info in enumerate(custom_expressions):
                name = expr_info.get("name", f"custom_{i}")
                expressions[name] = {
                    "name": name,
                    "type": "custom",
                    "is_binary": expr_info.get("isBinary", False)
                }

                print(f"Custom Expression: {name}")
                print(f"  Binary: {expr_info.get('isBinary', False)}")
                print()

    # Save to JSON
    output_file = "kurisu_vrm_expressions.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "vrm_version": vrm_version,
            "total_expressions": len(expressions),
            "expressions": expressions,
            "expression_names": list(expressions.keys())
        }, f, indent=2, ensure_ascii=False)

    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print(f"VRM Version: {vrm_version}")
    print(f"Total expressions: {len(expressions)}")
    print()
    print("Expression list:")
    for name in expressions.keys():
        print(f"  - {name}")
    print()
    print(f"✅ Expression data saved to: {output_file}")
    print()

    return expressions


if __name__ == "__main__":
    vrm_file = Path("static/models/Kurisu.vrm")

    if not vrm_file.exists():
        print(f"❌ VRM file not found: {vrm_file}")
        print("Please ensure Kurisu.vrm is in static/models/")
        exit(1)

    expressions = extract_vrm_expressions(str(vrm_file))
