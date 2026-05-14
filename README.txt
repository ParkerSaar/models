=== CS2 Multi-Agent Chams Bundle ===

CONTENTS:
  aim_trainer_3d__2_.as      — Updated script with multi-agent runtime
  glb_to_chams_bin.py        — GLB→.bin converter (for adding more agents later)
  chams_bins/                — All 60 converted .bin mesh files

INSTALLATION:
  1. Drop aim_trainer_3d__2_.as into your PCX scripts folder (replace existing)
  2. Drop ALL files from chams_bins/ into your PCX script directory (where the
     existing chams_mesh.bin and ctm_sas_mesh.bin live)
  3. Make sure your existing chams_mesh.bin (T phoenix default) is still there
  4. Restart the script in PCX

REGISTERED AGENTS:
  - 61 base agents (one slot per unique mesh)
  - ~30 alias names (variants that share meshes via texture-only differences,
    duplicates pointing to same .bin to save VRAM)
  - Total: ~90 CS2 model names mapped

LAZY LOADING:
  Only chams_mesh.bin (T) and ctm_sas_mesh.bin (CT) load at startup.
  Other agents download/load on first sighting in a match.
  Falls back to defaults if a download fails.

ADDING MORE AGENTS LATER:
  1. Extract GLB from CS2 via Source2Viewer
  2. Run: python3 glb_to_chams_bin.py input.glb output_mesh.bin --max-bones 94 (CT) or 86 (T)
  3. Add a register_agent() line in register_default_agents() in the script
  4. Drop the .bin in your script directory

BUMP NOTES:
  - CHAMS_MAX_AGENTS = 128 (room for 67 more)
  - Lazy-loaded; no startup VRAM cost for agents not in current match
