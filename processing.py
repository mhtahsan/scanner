import cv2
import numpy as np
from PIL import Image
from skimage.filters import threshold_sauvola
from skimage import img_as_ubyte
import streamlit as st

def find_document_corners(img):

    height, width = img.shape[:2]
    ImageArea = height * width
    minArea = ImageArea * 0.1

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 70, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:6]

    for c in contours:
        contour_area = cv2.contourArea(c)
        if contour_area < minArea:
            continue
        peri = cv2.arcLength(c, True)
        if peri < 100:
            continue
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
    return None

def fix_perspective(img, corners):
    ordered_corners = np.zeros((4, 2), dtype="float32")

    corner_sums = corners.sum(axis=1)
    ordered_corners[0] = corners[np.argmin(corner_sums)]  
    ordered_corners[2] = corners[np.argmax(corner_sums)]
    corner_diffs = np.diff(corners, axis=1)
    ordered_corners[1] = corners[np.argmin(corner_diffs)]  
    ordered_corners[3] = corners[np.argmax(corner_diffs)]

    (top_left, top_right, bottom_right, bottom_left) = ordered_corners

    width1 = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))
    width2 = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))
    max_width = max(int(width1), int(width2))
  
    height1 = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + ((top_right[1] - bottom_right[1]) ** 2))
    height2 = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))
    max_height = max(int(height1), int(height2))

    if max_width < 50 or max_height < 50:
        raise ValueError("Calculated dimensions too small")
    
    destination_corners = np.array([
        [0, 0],                     
        [max_width - 1, 0],                
        [max_width - 1, max_height - 1],     
        [0, max_height - 1]                  
    ], dtype="float32")

    transform_matrix = cv2.getPerspectiveTransform(ordered_corners, destination_corners)
    corrected_image = cv2.warpPerspective(img, transform_matrix, (max_width, max_height))
    
    return corrected_image


def clean_up_image(image):
    """
    Clean up the image: remove noise and enhance text
    """
    # Convert to grayscale if it's colored
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Remove noise from the image
    #denoised = cv2.fastNlMeansDenoising(gray)
     

    # Apply Sauvola thresholding for better text contrast
    # This works better than regular thresholding for documents
    #clean_image = sauvola_threshold(denoised)
    thresh = threshold_sauvola(gray, window_size=25)
    binary = gray > thresh
    output = img_as_ubyte(binary)
    
    inverse = 255 - output
    erode = cv2.erode(inverse, np.ones((2, 2), np.uint8))
    dilate = cv2.dilate(erode, np.ones((2, 2), np.uint8))
    clean_image = 255 - dilate
    return clean_image

def process_image(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    try:
        corners = find_document_corners(img)
        if corners is not None:
            try:
                corrected_image = fix_perspective(img, corners)
                st.success("✅ Document detected and perspective corrected!")
            except Exception as e:
                st.warning("⚠️ Document detected but perspective correction failed - using original image")
                corrected_image = img
        else:
            corrected_image = img
            st.info("ℹ️ No suitable document edges detected - using original image")
    except Exception as e:
        st.warning("⚠️ Document detection failed - using original image")
        corrected_image = img
    
    try:
        final_image = clean_up_image(corrected_image)
    except Exception as e:
        st.warning("⚠️ Advanced cleaning failed - using basic processing")
        if len(corrected_image.shape) == 3:
            final_image = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2GRAY)
        else:
            final_image = corrected_image
    
    # Convert back to PIL format for display and PDF creation
    pil_image = Image.fromarray(final_image)
    
    return pil_image

