"""
Team Player Assignment — CLIP zero-shot jersey colour classification
=====================================================================
For each detected player crop, CLIP compares the image against text
prompts like "a cricket player wearing a red jersey" vs
"a cricket player wearing a blue jersey" and picks the best match.

Public API used by player_detection.py:
    ask_team_colors()                              → list[str]
    load_clip_model()                              → (model, processor)
    classify_player_team(crop, model, processor,
                         team_colors)              → team_index (int)
    get_box_color(team_index, team_colors)         → BGR tuple
"""

import cv2
import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# CLIP model
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# Known colour name → OpenCV BGR value
COLOR_MAP_BGR = {
    "red":    (0,   0,   255),
    "blue":   (255, 0,   0  ),
    "green":  (0,   200, 0  ),
    "yellow": (0,   255, 255),
    "white":  (255, 255, 255),
    "black":  (30,  30,  30 ),
    "orange": (0,   165, 255),
    "purple": (128, 0,   128),
    "pink":   (203, 192, 255),
    "cyan":   (255, 255, 0  ),
    "navy":   (128, 0,   0  ),
    "maroon": (0,   0,   128),
    "gray":   (128, 128, 128),
    "grey":   (128, 128, 128),
    "brown":  (19,  69,  139),
    "gold":   (0,   215, 255),
    "teal":   (128, 128, 0  ),
    "lime":   (0,   255, 0  ),
}

# Fallback colours when a jersey name has no entry in COLOR_MAP_BGR
FALLBACK_COLORS = [
    (0,   255, 0  ),   # green
    (255, 0,   0  ),   # blue
    (0,   0,   255),   # red
    (0,   255, 255),   # yellow
    (255, 0,   255),   # magenta
]

# Public API

def ask_team_colors():
    """
    Prompt the user for jersey colours at startup.

    Example input:  blue, red
                    yellow, green, white

    Returns a list of lowercase colour strings, one per team.
    """
    print("\n" + "=" * 55)
    print("  TEAM JERSEY COLOUR SETUP")
    print("  Enter each team's jersey colour separated by commas.")
    print("  Example:  blue, red   |   yellow, green")
    print("=" * 55)
    raw = input("  Jersey colours: ").strip()
    colors = [c.strip().lower() for c in raw.split(",") if c.strip()]

    if len(colors) < 2:
        print("  [Warning] Need at least 2 colours. Defaulting to: red, blue")
        colors = ["red", "blue"]

    print(f"  Teams set → {' | '.join(f'Team {i+1}: {c}' for i, c in enumerate(colors))}")
    print("=" * 55 + "\n")
    return colors


def load_clip_model():
    """Download / load CLIP and return (model, processor)."""
    print("  [CLIP] Loading model...")
    model     = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model.eval()
    print("  [CLIP] Ready.")
    return model, processor


def classify_player_team(crop_bgr, model, processor, team_colors,
                         min_confidence=0.6):
    """
    Zero-shot classify which team the player belongs to.

    Parameters
    ----------
    crop_bgr       : np.ndarray  — player crop in BGR (from OpenCV)
    model          : CLIPModel
    processor      : CLIPProcessor
    team_colors    : list[str]   — e.g. ["red", "blue"]
    min_confidence : float       — if the best match probability is below
                                   this threshold the player is considered
                                   "unknown" and -1 is returned so no
                                   bounding box is drawn.

    Returns
    -------
    int  team index (0-based), or -1 if crop is invalid or no team colour
         matches confidently enough.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return -1

    # Build one text prompt per team plus an "other" catch-all prompt
    team_texts = [
        f"a cricket player wearing a {color} jersey"
        for color in team_colors
    ]
    other_text = ["a person wearing a different colored jersey"]
    all_texts  = team_texts + other_text

    # BGR → PIL RGB
    pil_img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))

    inputs = processor(
        text=all_texts,
        images=pil_img,
        return_tensors="pt",
        padding=True,
    )

    with torch.no_grad():
        logits = model(**inputs).logits_per_image   # shape: (1, n_teams+1)
        probs  = logits.softmax(dim=1)[0]           # shape: (n_teams+1,)

    # Only consider team prompts (exclude the "other" entry at the end)
    team_probs  = probs[:-1]
    best_idx    = int(team_probs.argmax().item())
    best_conf   = float(team_probs[best_idx].item())

    # If best team confidence is below threshold → unknown jersey, skip
    if best_conf < min_confidence:
        return -1

    return best_idx


def get_box_color(team_idx, team_colors):
    """
    Return the BGR bounding-box colour for a given team index.
    Falls back gracefully for unknown colour names.
    """
    if team_idx < 0 or team_idx >= len(team_colors):
        return (0, 255, 0)   # default green

    color_name = team_colors[team_idx].lower()

    if color_name in COLOR_MAP_BGR:
        return COLOR_MAP_BGR[color_name]

    # Not in the map — use a distinct fallback colour
    return FALLBACK_COLORS[team_idx % len(FALLBACK_COLORS)]
