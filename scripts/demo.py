import argparse
import os
import os.path as osp
import numpy as np
import cv2
import torch
import gc
import sys
import colorsys
import shutil


# Add the path to SAM2 module
sys.path.append("./sam2")
from sam2.build_sam import build_sam2_video_predictor


def gen_col(N):
    """
    Generate a dictionary of N distinguishable colors in RGB format,
    indexed from 0 to N-1.
    """
    colors = {}
    for i in range(N):
        hue = i / N
        saturation = 1.0
        value = 1.0
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        rgb_normalized = tuple(int(x * 255) for x in rgb)
        colors[i] = rgb_normalized
    return colors


def safe_propagate_in_video(predictor, state):
    """
    Generator that wraps predictor.propagate_in_video(state)
    and ignores KeyError if it arises, skipping that frame.
    """
    gen = predictor.propagate_in_video(state)
    while True:
        try:
            yield next(gen)
        except StopIteration:
            break
        except KeyError as e:
            print(f"[WARNING] Caught KeyError: {e}. Skipping this frame.")
            continue


# --------------------------------------------------------------------------------
# 1) Helper functions
# --------------------------------------------------------------------------------

def set_color(num_colors):
    """
    Generate a list of distinct, vivid colors (RGB tuples) by varying the hue.
    """
    colors = []
    for i in range(num_colors):
        h = i / num_colors
        s = 1.0  # full saturation for vividness
        v = 1.0  # full value for brightness
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


def determine_model_cfg(model_path):
    """
    Pick the correct SAM2 config file based on keywords in the model path.
    """
    if "large" in model_path:
        return "configs/boxtrack/sam2.1_hiera_l.yaml"
    elif "base_plus" in model_path:
        return "configs/boxtrack/sam2.1_hiera_b+.yaml"
    elif "small" in model_path:
        return "configs/boxtrack/sam2.1_hiera_s.yaml"
    elif "tiny" in model_path:
        return "configs/boxtrack/sam2.1_hiera_t.yaml"
    # else:
    #     # default fallback
    #     return "configs/boxtrack/sam2.1_hiera_b+.yaml"


def create_intervals(total_frames, mode, param):
    """
    (ORIGINAL FUNCTION)
    Creates a list of (start_frame, end_frame) intervals based on:
      - mode = 'interval': param is a chunk size (int).
      - mode = 'first_app': param is a sorted list of frame boundaries.
      - mode = None or invalid: single interval [1..total_frames].
    """
    if not mode:
        return [(1, total_frames)]

    intervals = []
    if mode == "interval":
        chunk_size = param
        start = 1
        while start <= total_frames:
            end = min(start + chunk_size - 1, total_frames)
            intervals.append((start, end))
            start = end + 1

    elif mode == "first_app":
        boundaries = param  # e.g. [20, 34, 50, 120, 219]
        prev = 1
        for boundary in boundaries:
            if boundary < prev:
                continue
            boundary = min(boundary, total_frames)
            intervals.append((prev, boundary))
            prev = boundary + 1
            if prev > total_frames:
                break
        if prev <= total_frames:
            intervals.append((prev, total_frames))
    else:
        # If something unexpected, just return the whole range
        intervals = [(1, total_frames)]

    return intervals


def create_intervals_first_app_and_interval(total_frames, boundaries, chunk_size):
    """
    1) Splits [1..total_frames] using the 'boundaries' list in first_app style.
    2) For each of those intervals, if it exceeds chunk_size, subdivide into
       consecutive chunks of 'chunk_size'.

    Parameters
    ----------
    total_frames : int
        Total number of frames in the video (e.g., 958).
    boundaries : list of int
        Sorted list of frame boundaries where new intervals should end
        (read from man_track_file). e.g. [20, 200, 260, 500, 958]
    chunk_size : int
        The maximum length of any sub-interval, e.g. 100.

    Returns
    -------
    final_intervals : list of (start_frame, end_frame)
        A list of 1-based inclusive intervals, e.g. [(1,20), (21,100), (101,120), ...]
    """
    # 1. Create intervals from [1..total_frames] using first_app style
    first_app_intervals = []
    prev = 1
    for boundary in boundaries:
        if boundary < prev:
            continue
        # cap the boundary if it goes beyond total_frames
        boundary = min(boundary, total_frames)
        first_app_intervals.append((prev, boundary))
        prev = boundary + 1
        if prev > total_frames:
            break

    if prev <= total_frames:
        first_app_intervals.append((prev, total_frames))

    # 2. Now subdivide each of those intervals if it exceeds chunk_size
    final_intervals = []
    for (start_f, end_f) in first_app_intervals:
        length = end_f - start_f + 1
        if length <= chunk_size:
            final_intervals.append((start_f, end_f))
        else:
            # break it into consecutive chunks of 'chunk_size'
            current_start = start_f
            while current_start <= end_f:
                sub_end = min(current_start + chunk_size - 1, end_f)
                final_intervals.append((current_start, sub_end))
                current_start = sub_end + 1

    return final_intervals


def get_starting_frames(filename):
    """
    Example function to read 'man_track_file' lines and
    collect the second field as a frame boundary.
    """
    with open(filename, 'r') as f:
        frames = []
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                frames.append(int(parts[1]))
            except ValueError:
                continue
    
    # example logic: sorted unique, and optionally remove the first index if needed
    final_frame = sorted(set(frames))
    # final_frame.pop(0)  # <-- depends on your data
    return final_frame


def sort_gt_lines(gt_lines):
    def sort_key(line):
        parts = line.strip().split(',')
        try:
            obj_id = int(parts[1])
            frame_id = int(parts[0])
        except (IndexError, ValueError):
            obj_id, frame_id = 0, 0
        return (obj_id, frame_id)
    
    return sorted(gt_lines, key=sort_key)


def filter_gt_for_interval(input_file, start_frame, end_frame, temp_gt_file):
    """
    Reads input_file (MOT-like or custom: frame_id, object_id, x, y, w, h, ...)
    and writes only lines where frame_id in [start_frame..end_frame]
    to temp_gt_file. Returns the max object_id found (unused here).
    """
    max_obj_id = 0
    with open(input_file, 'r') as infile, open(temp_gt_file, 'w') as outfile:
        for line in infile:
            parts = line.strip().split(',')
            frame_id = int(parts[0])
            if start_frame <= frame_id <= end_frame:
                object_id = int(parts[1])
                max_obj_id = max(max_obj_id, object_id)
                outfile.write(line)
    return max_obj_id


def copy_frames_for_interval(video_path, start_frame, end_frame, temp_folder, all_frames_list=None):
    """
    Copies frames [start_frame..end_frame] to a temporary folder.
    If 'video_path' is a directory, we assume 'all_frames_list' is a sorted
    list of filenames. If it's .mp4, we read from OpenCV and write images out.
    """
    if osp.exists(temp_folder):
        shutil.rmtree(temp_folder)
    os.makedirs(temp_folder, exist_ok=True)

    if osp.isdir(video_path):
        if not all_frames_list:
            raise ValueError("all_frames_list is required for frame directories.")

        for idx in range(start_frame, end_frame + 1):
            frame_name = all_frames_list[idx - 1]  # because idx is 1-based
            src = osp.join(video_path, frame_name)
            local_idx = idx - start_frame + 1
            dst = osp.join(temp_folder, f"{local_idx:06d}.jpg")
            shutil.copy2(src, dst)

    else:
        # .mp4 -> read via OpenCV, write each relevant frame
        cap = cv2.VideoCapture(video_path)
        current_frame = 0
        written_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            current_frame += 1
            if start_frame <= current_frame <= end_frame:
                written_count += 1
                filename = f"{written_count:06d}.jpg"
                cv2.imwrite(osp.join(temp_folder, filename), frame)
            if current_frame >= end_frame:
                break
        cap.release()

    return osp.abspath(temp_folder)


def gt_to_entry(input_file, output_file):
    """
    Extracts the first bounding box of each object from input_file,
    writes it to output_file, and returns a dict:
       fid -> (frame_id, ((x1,y1,x2,y2), 0))
    """
    first_appearance = {}
    with open(input_file, 'r') as infile:
        for line in infile:
            parts = line.strip().split(',')
            frame_id = int(parts[0])
            object_id = int(parts[1])
            x_str, y_str, w_str, h_str = parts[2:6]
            if object_id not in first_appearance:
                first_appearance[object_id] = (frame_id, (x_str, y_str, w_str, h_str))

    with open(output_file, 'w') as outfile:
        for _, (fid, bbox) in first_appearance.items():
            x_str, y_str, w_str, h_str = bbox
            outfile.write(f"{x_str},{y_str},{w_str},{h_str}\n")

    prompts = {}
    for fid, (frame_id, (x_str, y_str, w_str, h_str)) in enumerate(first_appearance.values()):
        x, y, w, h = map(float, [x_str, y_str, w_str, h_str])
        x, y, w, h = int(x), int(y), int(w), int(h)
        prompts[fid] = (frame_id, ((x, y, x + w, y + h), 0))

    return prompts

def post_process_results(
    pred_gt,
    frames_to_save,
    frames,
    num_objects,
    plot_option,
    output_tracking_txt,
    output_dir,
    min_bbox_side=0
):
    """
    1) Saves the tracking data to 'output_tracking_txt' in MOT-like format.
    2) Visualizes the results by drawing:
         - Segmentation masks (with a semi-transparent fill and a border),
         - Bounding boxes with a fixed thickness,
         - Labels that always stick to the bounding box (above it if possible, otherwise below),
       and saves the resulting frames in 'output_dir'.
    """
    import os
    import cv2
    import numpy as np
    import gc

    # --- Save the unified tracking results ---
    formatted_data = []
    for (g_obj_idx, g_frame_idx, (x, y, w, h), confidence) in pred_gt:
        if w < min_bbox_side or h < min_bbox_side:
            continue
        frame_id = g_frame_idx + 1
        object_id = g_obj_idx + 1
        conf_str = f"{confidence:.4f}"
        line = f"{frame_id},{object_id},{x},{y},{w},{h},{conf_str},-1,-1,-1"
        formatted_data.append(line)

    formatted_data = sort_gt_lines(formatted_data)
    with open(output_tracking_txt, "w") as f:
        f.write("\n".join(formatted_data))
    print(f"Tracking data saved to {output_tracking_txt}")

    # --- Skip visualization if no frames are provided ---
    if not frames:
        print("No frames were loaded or '--save_to_video' was not set. Skipping visualization.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # --- Generate a fixed palette of 25 colors using your provided function ---
    color_map = gen_col(25)

    # --- Process each frame to draw visualizations ---
    for f_idx, obj_dict in frames_to_save.items():
        if f_idx < 0 or f_idx >= len(frames):
            continue
        img = frames[f_idx].copy()
        height, width = img.shape[:2]

        for g_obj_idx, (mask_dict, bbox_dict) in obj_dict.items():
            # Choose a color from the fixed palette.
            color = color_map[g_obj_idx % 25]

            # ----- Draw segmentation masks and borders -----
            if plot_option in ["mask", "both"]:
                for _, mask in mask_dict.items():
                    # Create an overlay filled with the object color.
                    overlay = np.zeros((height, width, 3), dtype=np.uint8)
                    overlay[mask] = color
                    img = cv2.addWeighted(img, 1, overlay, 0.3, 0)

                    # Find contours and draw a border around the segmentation.
                    mask_uint8 = (mask.astype(np.uint8)) * 255
                    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        cv2.drawContours(img, contours, -1, color, thickness=1)

            # ----- Draw bounding boxes and labels -----
            if plot_option in ["bbox", "both"] or plot_option == "mask":
                for _, bbox_arr in bbox_dict.items():
                    bx, by, bw, bh = bbox_arr
                    if bw < min_bbox_side or bh < min_bbox_side:
                        continue

                    # Draw the bounding box with a thickness of 2.
                    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), color, thickness=2)

                    # Prepare the label (object ID) text.
                    label_text = str(g_obj_idx + 1)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    font_scale = 0.4
                    text_thickness = 1
                    (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, text_thickness)

                    # Determine vertical placement:
                    # Default: above the bounding box if there's enough space.
                    if by - (text_height + baseline) >= 0:
                        label_top = by - (text_height + baseline)
                    else:
                        # Not enough room above; place the label immediately below the box.
                        label_top = by + bh

                    # Determine horizontal placement (stick to left of bounding box).
                    label_left = bx
                    # If the label would go beyond the image's right edge, adjust it.
                    if label_left + text_width > width:
                        label_left = width - text_width

                    # Draw a filled rectangle as background for the label.
                    cv2.rectangle(
                        img,
                        (label_left, label_top),
                        (label_left + text_width, label_top + text_height + baseline),
                        color,
                        thickness=cv2.FILLED
                    )
                    # Draw the label text in white.
                    cv2.putText(
                        img,
                        label_text,
                        (label_left, label_top + text_height),
                        font,
                        font_scale,
                        (255, 255, 255),
                        thickness=text_thickness,
                        lineType=cv2.LINE_AA
                    )

        # Save the annotated frame.
        save_path = os.path.join(output_dir, f"{f_idx+1:06d}.jpg")
        cv2.imwrite(save_path, img)

    print("Visualization images saved to", output_dir)
    gc.collect()


# --------------------------------------------------------------------------------
# 3) Main function
# --------------------------------------------------------------------------------

def main(args):
    """
    Runs SAM2 on intervals, merges results, and optionally visualizes them.
    The crucial fix is always using obj_id=0 inside SAM2 to avoid KeyError.
    """
    # 1. Build predictor
    model_cfg = determine_model_cfg(args.model_path)
    print("Using config:", model_cfg)
    predictor = build_sam2_video_predictor(model_cfg, args.model_path, device="cuda:0")

    # 2. Determine total frames
    if osp.isdir(args.video_path):
        all_frames_list = sorted([
            f for f in os.listdir(args.video_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        total_frames = len(all_frames_list)
    else:
        cap = cv2.VideoCapture(args.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    print(f"Found total_frames={total_frames}")

    # 3. Figure out the intervals to process
    intervals = []
    if args.interval_mode == "combined":
        # Step A: read boundaries from man_track_file
        if not args.man_track_file:
            raise ValueError("Please provide --man_track_file when using interval_mode='combined'")
        # boundaries = []
        boundaries = get_starting_frames(args.man_track_file)
        boundaries.pop(0)
        # Step B: subdivide large intervals > chunk_size
        intervals = create_intervals_first_app_and_interval(total_frames, boundaries, args.interval_param)

    elif args.interval_mode == "first_app":
        raw_intervals = get_starting_frames(args.man_track_file)  # returns something like [20, 34, 50, ...]
        intervals = create_intervals(total_frames, "first_app", raw_intervals)

    elif args.interval_mode == "interval":
        intervals = create_intervals(total_frames, "interval", args.interval_param)

    else:
        # Fallback: single interval from 1..total_frames
        intervals = [(1, total_frames)]

    print("Intervals:", intervals)

    # 4. We'll store final results here
    full_pred_gt = []        # list of (global_obj_idx, global_frame_idx, (x,y,w,h), confidence)
    full_frames_to_save = {} # dict: global_frame_idx -> { global_obj_idx: [mask_dict, bbox_dict] }
    total_objects_so_far = 0 # how many objects so far

    # If the input is a folder, keep the list for copying frames
    all_frames_list = sorted(all_frames_list) if osp.isdir(args.video_path) else None

    # 5. Process intervals
    if not intervals:
        intervals = [(1, total_frames)]

    for i_idx, (start_f, end_f) in enumerate(intervals):
        print(f"\n[Interval {i_idx}]: frames={start_f}..{end_f}")

        # A) filter GT for this interval
        temp_gt_file = f"temp_gt_{i_idx}.txt"
        filter_gt_for_interval(args.txt_path, start_f, end_f, temp_gt_file)

        # B) Get sub-prompts
        temp_output_file = f"temp_output_{i_idx}.txt"
        sub_prompts = gt_to_entry(temp_gt_file, temp_output_file)
        n_objects_here = len(sub_prompts)
        print(f"  Found {n_objects_here} new object(s) in this interval.")

        # C) Copy frames to temp folder
        temp_frames_folder = f"temp_frames_{i_idx}"
        copy_frames_for_interval(args.video_path, start_f, end_f,
                                 temp_frames_folder, all_frames_list)

        # D) For each object in sub_prompts, run tracker with obj_id=0
        partial_pred_gt = []
        partial_frames_to_save = {}

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            for key, value in sub_prompts.items():
                orig_frame_id, (bbox_coords, _) = value
                local_start_idx = (orig_frame_id - start_f)  # 0-based index

                # Re-init state for each object
                state = predictor.init_state(temp_frames_folder, offload_video_to_cpu=True)

                # Pass obj_id=0 to avoid KeyError
                _, _, init_masks = predictor.add_new_points_or_box(
                    state,
                    box=bbox_coords,
                    frame_idx=local_start_idx,
                    obj_id=0
                )

                # Propagate
                for (local_fr, object_ids, masks, score) in safe_propagate_in_video(predictor, state):
                    object_score = torch.sigmoid(score).item()

                    mask_dict = {}
                    bbox_dict = {}
                    for obj_id, m in zip(object_ids, masks):
                        m_numpy = (m[0].cpu().numpy() > 0.0)
                        mask_dict[obj_id] = m_numpy
                        nz = np.argwhere(m_numpy)
                        if len(nz) == 0:
                            b_res = [0, 0, 0, 0]
                        else:
                            y_min, x_min = nz.min(axis=0)
                            y_max, x_max = nz.max(axis=0)
                            b_res = [x_min, y_min, (x_max - x_min), (y_max - y_min)]
                        bbox_dict[obj_id] = b_res

                    global_fr_idx = (start_f - 1) + local_fr
                    if global_fr_idx not in partial_frames_to_save:
                        partial_frames_to_save[global_fr_idx] = {}
                    partial_frames_to_save[global_fr_idx][key] = [mask_dict, bbox_dict]

                    partial_pred_gt.append([
                        key,
                        global_fr_idx,
                        b_res,
                        object_score
                    ])

        # E) Merge partial results into global
        for fr_idx, obj_map in partial_frames_to_save.items():
            if fr_idx not in full_frames_to_save:
                full_frames_to_save[fr_idx] = {}
            for g_obj_idx, val in obj_map.items():
                full_frames_to_save[fr_idx][g_obj_idx] = val

        full_pred_gt.extend(partial_pred_gt)
        total_objects_so_far += n_objects_here

        # Cleanup temp files/folders
        if osp.exists(temp_gt_file):
            os.remove(temp_gt_file)
        if osp.exists(temp_output_file):
            os.remove(temp_output_file)
        shutil.rmtree(temp_frames_folder, ignore_errors=True)

    # 6. Final post-processing
    print("\nDone with intervals. Creating unified output...")

    # If user wants annotation, load all frames
    all_frames = []
    if args.save_to_video:
        print("Loading all frames for final visualization...")
        if osp.isdir(args.video_path):
            # Directory of frames
            frame_files = sorted([
                osp.join(args.video_path, f)
                for f in os.listdir(args.video_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])
            all_frames = [cv2.imread(fp) for fp in frame_files]
        else:
            # .mp4 video
            cap = cv2.VideoCapture(args.video_path)
            while True:
                ret, frm = cap.read()
                if not ret:
                    break
                all_frames.append(frm)
            cap.release()

    post_process_results(
        pred_gt=full_pred_gt,
        frames_to_save=full_frames_to_save,
        frames=all_frames,
        num_objects=total_objects_so_far,
        plot_option=args.plot_option,
        output_tracking_txt=args.output_tracking_txt,
        output_dir=args.output_dir,
        min_bbox_side=args.min_bbox_side
    )

    print("All done!")
    del predictor
    gc.collect()
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True,
                        help="Path to either .mp4 or a directory of frames.")
    parser.add_argument("--txt_path", required=True,
                        help="MOT-like GT file: frame_id, object_id, x, y, w, h,...")
    parser.add_argument("--model_path", default="sam2/checkpoints/sam2.1_hiera_large.pt",
                        help="Path to the SAM2 model checkpoint.")
    parser.add_argument("--output_tracking_txt", required=True,
                        help="Output .txt for final (unified) tracking results.")
    parser.add_argument("--output_dir", default="./output_images",
                        help="Directory to save annotated frames.")
    parser.add_argument("--save_to_video", action="store_true",
                        help="If set, loads frames to produce annotated images.")
    parser.add_argument("--plot_option", default="both", choices=["bbox", "mask", "both"],
                        help="How to display results on the frames.")
    parser.add_argument("--min_bbox_side", type=int, default=0,
                        help="Skip bounding boxes smaller than this side length.")

    # New or existing mode choices:
    parser.add_argument("--interval_mode", default=None,
                        choices=[None, "interval", "first_app", "combined"],
                        help="Split frames by chunk size (interval), a boundary list (first_app), or do both (combined).")
    parser.add_argument("--interval_param", default="100",
                        help="If 'interval', chunk size as an int. "
                             "If 'first_app' or 'combined', a comma-separated list or int chunk size. "
                             "For 'combined', used as chunk_size in the second step.")
    parser.add_argument("--man_track_file", default=None,
                        help="Path to a file with custom boundaries (for first_app or combined).")

    args = parser.parse_args()

    # Convert interval_param if needed
    if args.interval_mode == "interval":
        args.interval_param = int(args.interval_param)
    elif args.interval_mode == "first_app":
        # parse comma-separated boundaries
        parts = args.interval_param.split(",")
        args.interval_param = [int(x.strip()) for x in parts if x.strip().isdigit()]
    elif args.interval_mode == "combined":
        # if 'combined', we use 'interval_param' as the chunk_size integer
        # boundaries come from man_track_file
        args.interval_param = int(args.interval_param)

    main(args)
