import cv2

from utils.image_processing import apply_enhancement


def process_frame(frame, model, enhancement_type, region_coords, conf, counted_ids):
    """
    Core function: enhancement -> tracking -> region filtering -> unique counting.
    The 'counted_ids' input is a set used to store unique object IDs.
    """
    enhanced_frame = apply_enhancement(frame, enhancement_type)
    results = model.track(enhanced_frame, conf=conf, persist=True, verbose=False)

    rx1, ry1, rx2, ry2 = region_coords

    overlay = enhanced_frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, enhanced_frame, 0.7, 0, enhanced_frame)

    roi = frame[ry1:ry2, rx1:rx2]
    if roi.size > 0:
        enhanced_frame[ry1:ry2, rx1:rx2] = apply_enhancement(roi, enhancement_type)

    cv2.rectangle(enhanced_frame, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        clss = results[0].boxes.cls.int().cpu().tolist()

        for box, track_id, cls in zip(boxes, track_ids, clss):
            x1, y1, x2, y2 = map(int, box)
            label = model.names[cls]

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            if (rx1 < center_x < rx2) and (ry1 < center_y < ry2):
                if track_id not in counted_ids:
                    counted_ids.add(track_id)

                cv2.rectangle(enhanced_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(enhanced_frame, (center_x, center_y), 5, (0, 0, 255), -1)
                cv2.putText(
                    enhanced_frame,
                    f"#{track_id} {label}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

    total_count = len(counted_ids)

    cv2.putText(
        enhanced_frame,
        f"DETECTED: {total_count}",
        (10, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2,
    )
    cv2.putText(
        enhanced_frame,
        "Yellow Box: Active Region",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 255, 255),
        1,
    )

    return enhanced_frame, total_count
