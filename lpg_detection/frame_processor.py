import cv2

from lpg_detection.enhancement import apply_enhancement


def process_frame(frame, model, enhancement_type, region_coords, conf, counted_ids):
    """
    Core function: enhancement -> tracking -> region filtering -> unique counting.
    The 'counted_ids' input is a set used to store unique object IDs.
    """
    # 1. Apply enhancement.
    enhanced_frame = apply_enhancement(frame, enhancement_type)

    # 2. Run tracking. persist=True keeps object IDs stable across frames.
    # Uses the default tracker, such as 'bytetrack.yaml' or 'botsort.yaml'.
    results = model.track(enhanced_frame, conf=conf, persist=True, verbose=False)

    # Get region coordinates.
    rx1, ry1, rx2, ry2 = region_coords

    # --- REGION & BACKGROUND VISUALIZATION ---
    # Darken the area outside the active region for focus.
    overlay = enhanced_frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, enhanced_frame, 0.7, 0, enhanced_frame)

    # Restore brightness inside the active region.
    roi = frame[ry1:ry2, rx1:rx2]
    # Ensure the coordinates are valid.
    if roi.size > 0:
        enhanced_frame[ry1:ry2, rx1:rx2] = apply_enhancement(roi, enhancement_type)

    # Draw the active region boundary in yellow.
    cv2.rectangle(enhanced_frame, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)

    # 3. Iterate through each detected object.
    # Ensure detected objects exist and boxes are not empty.
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()  # Get tracking IDs.
        clss = results[0].boxes.cls.int().cpu().tolist()

        for box, track_id, cls in zip(boxes, track_ids, clss):
            x1, y1, x2, y2 = map(int, box)
            label = model.names[cls]

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # REGION FILTERING LOGIC
            # Objects are processed only if their center point is inside the active region.
            if (rx1 < center_x < rx2) and (ry1 < center_y < ry2):

                # Unique ID check: add it if it has not been counted before.
                if track_id not in counted_ids:
                    counted_ids.add(track_id)

                # Draw the counted bounding box in green.
                cv2.rectangle(enhanced_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(
                    enhanced_frame,
                    (center_x, center_y),
                    5,
                    (0, 0, 255),
                    -1,
                )  # Red center point.

                # Display label and ID.
                cv2.putText(
                    enhanced_frame,
                    f"#{track_id} {label}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            # Objects outside the active region are not drawn.

    # Count total unique IDs that have entered the active region.
    total_count = len(counted_ids)

    # ==========================================
    # LEGEND
    # ==========================================

    # 1. Detection count text in red.
    cv2.putText(enhanced_frame, f"DETECTED: {total_count}", (10, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 2. Region legend text in yellow.
    cv2.putText(enhanced_frame, "Yellow Box: Active Region", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    return enhanced_frame, total_count
