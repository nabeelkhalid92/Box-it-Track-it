#!/bin/bash

SCRIPT_PATH="python scripts/demo.py"
VIDEO_PATH="demo/demo_1/images"
TXT_PATH="demo/demo_1/bbox.txt"
MODEL_PATH="sam2/checkpoints/sam2.1_hiera_large.pt"
OUTPUT_TRACKING_TXT="demo/demo_1/tracking_results.txt"
OUTPUT_DIR="demo/demo_1/output"

$SCRIPT_PATH \
    --video_path "$VIDEO_PATH" \
    --txt_path "$TXT_PATH" \
    --model_path "$MODEL_PATH" \
    --output_tracking_txt "$OUTPUT_TRACKING_TXT" \
    --output_dir "$OUTPUT_DIR" \
    --plot_option both \
    --save_to_video \
    --min_bbox_side 5 \
    --interval_mode interval \
    --interval_param 300 \
    --man_track_file "$MAN_TRACK_FILE"
