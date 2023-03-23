import cv2

def findContours(original_image, thresholded_image, min_Area=1000, sort=True, filter=0,drawCon=True,c=(255, 0, 0)):
    """
    Find and draw the contours in an Image.
    :param original_image: Image on which we want to draw.
    :param thresholded_image: Image on which we want to find contours.
    :param min_Area: Minimum Area to detect a valid contour.
    :param sort: This will sort the contours based on area.
    :param filter: Filter based on corner points.
    :Param drawCon: Boolean indicating whether to draw contours.
    :return: Image after drawing contours, contours with stats.
    """
    contoursFound = []
    imgContours = original_image.copy()
    contours, hierarchy = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Iterate through all the found contours.
    for contour in contours:
        area = cv2.contourArea(contour) # Area of current contour.
        if area > min_Area:
            # Compute the approximate contour Points.
            epsilon = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * epsilon, True)

            if len(approx) == filter or filter == 0:
                if drawCon:
                    cv2.drawContours(imgContours, contour, -1, c, 3) # Draw the current contour.
                    x, y, w, h = cv2.boundingRect(approx) # rec coordinates of curr contour.
                    cx, cy = x + (w // 2), y + (h // 2)
                    cv2.rectangle(imgContours, (x, y), (x + w, y + h), c, 2) # Draw rectangle around contour.
                    cv2.circle(imgContours, (x + (w // 2), y + (h // 2)), 5, c, cv2.FILLED) # Draw circle around contour.
                    contoursFound.append({"contour": contour, "area": area, "bbox": [x, y, w, h], "center": [cx, cy]})

    if sort: # Sort based on area.
        contoursFound = sorted(contoursFound, key=lambda x: x["area"], reverse=True)

    return imgContours, contoursFound
