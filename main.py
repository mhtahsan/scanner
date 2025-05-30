import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
import tempfile
import os
from processing import clean_up_image, process_image, fix_perspective, find_document_corners


def create_pdf_from_images(image_list):
    """
    Create a PDF file from a list of images
    """
    # Create a temporary file for the PDF
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    
    # Create the PDF
    pdf_canvas = canvas.Canvas(temp_pdf.name, pagesize=A4)
    
    # Add each image to the PDF
    for img in image_list:
        # Convert PIL image to bytes so we can add it to PDF
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Create an image reader for the PDF
        img_reader = ImageReader(img_bytes)
        
        # Get image dimensions
        img_width, img_height = img.size
        
        # Get PDF page dimensions
        page_width, page_height = A4
        
        # Scale the image to fit the page
        scale_factor = min(page_width / img_width, page_height / img_height)
        scaled_width = img_width * scale_factor
        scaled_height = img_height * scale_factor
        
        # Center the image on the page
        x_position = (page_width - scaled_width) / 2
        y_position = (page_height - scaled_height) / 2
        
        # Add the image to the PDF page
        pdf_canvas.drawImage(img_reader, x_position, y_position, 
                           width=scaled_width, height=scaled_height)
        
        # Start a new page for the next image
        pdf_canvas.showPage()
    
    # Save the PDF
    pdf_canvas.save()
    
    return temp_pdf.name

def move_image_up(image_list, index):
    """
    Move an image up in the list (swap with the one above it)
    """
    if index > 0:
        # Swap the current image with the one above it
        image_list[index], image_list[index - 1] = image_list[index - 1], image_list[index]
    return image_list

def move_image_down(image_list, index):
    """
    Move an image down in the list (swap with the one below it)
    """
    if index < len(image_list) - 1:
        # Swap the current image with the one below it
        image_list[index], image_list[index + 1] = image_list[index + 1], image_list[index]
    return image_list

def remove_image(image_list, index):
    """
    Remove an image from the list
    """
    if 0 <= index < len(image_list):
        image_list.pop(index)
    return image_list

def main():
    st.set_page_config(page_title="Image to PDF Converter", layout="wide")
    st.title("Image to PDF Converter")
    st.write("Upload images → They get processed automatically → Reorder them → Download as PDF!")

    if 'processed_images' not in st.session_state:
        st.session_state.processed_images = []
    if 'original_images' not in st.session_state:
        st.session_state.original_images = []

    st.subheader("Upload Images")
    uploaded_files = st.file_uploader(
        "Choose image files",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="You can select multiple images at once"
    )
    if uploaded_files:
        for uploaded_file in uploaded_files:
            existing_names = [name for _, name in st.session_state.original_images]
            if uploaded_file.name not in existing_names:
                file_size = uploaded_file.size
                original_image = Image.open(uploaded_file)
                width, height = original_image.size  # <-- Get size from PIL image
                if file_size > 5 * 1024 * 1024:
                    st.warning('Image is larger than 5MB, resizing...')
                    original_image = original_image.resize((width // 2, height // 2), Image.LANCZOS)
                st.session_state.original_images.append((original_image, uploaded_file.name))
                st.write(f"🔄 Processing: {uploaded_file.name}")
                processed_image = process_image(original_image)
                st.session_state.processed_images.append((processed_image, uploaded_file.name))


    if st.session_state.processed_images:
        st.subheader('Review and Reorder Images')
        for i, (processed_img, filename) in enumerate(st.session_state.processed_images):
            st.write("---")
            col1, col2, col3 = st.columns([1, 3, 1])
            with col1:
                st.write(f'**{filename}**')
                if i > 0:
                    if st.button('⬆️ Move Up', key=f'move_up_{i}'):
                        move_image_up(st.session_state.processed_images, i)
                        move_image_up(st.session_state.original_images, i)
                        st.rerun()
                if i < len(st.session_state.processed_images) - 1:  
                    if st.button("⬇️ Move Down", key=f"down_{i}"):
                        move_image_down(st.session_state.processed_images, i)
                        move_image_down(st.session_state.original_images, i)
                        st.rerun()  
                if st.button("🗑️ Remove", key=f"remove_{i}"):
                    remove_image(st.session_state.processed_images, i)
                    st.rerun()
                    remove_image(st.session_state.original_images, i)
                    st.rerun()
            with col2:
                st.image(processed_img, caption=f"Processed: {filename}", use_container_width=True)
            with col3:
                original_img = st.session_state.original_images[i][0]
                st.image(original_img, 
                        caption="Original", 
                        width=150)
       # PDF generation section
        st.write("---")
        st.subheader("📥 Step 3: Generate Your PDF")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 Create PDF", type="primary", use_container_width=True):
                if st.session_state.processed_images:
                    # Show progress
                    with st.spinner("Creating your PDF..."):
                        # Extract just the images (not the filenames)
                        images_only = [img for img, _ in st.session_state.processed_images]
                        
                        # Create the PDF
                        pdf_file_path = create_pdf_from_images(images_only)
                        
                        # Read the PDF file
                        with open(pdf_file_path, 'rb') as pdf_file:
                            pdf_data = pdf_file.read()
                        
                        # Clean up the temporary file
                        os.unlink(pdf_file_path)
                        
                        # Store PDF data for download
                        st.session_state.pdf_data = pdf_data
                        st.success("✅ PDF created successfully!")
        
        with col2:
            # Download button (only show if PDF was created)
            if 'pdf_data' in st.session_state:
                st.download_button(
                    label="📥 Download PDF",
                    data=st.session_state.pdf_data,
                    file_name="scanned_document.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        
        # Clear all button
        st.write("---")
        if st.button("🗑️ Clear All Images", type="secondary"):
            st.session_state.processed_images = []
            st.rerun()
            st.session_state.original_images = []
            st.rerun()
            if 'pdf_data' in st.session_state:
                del st.session_state.pdf_data
            st.rerun()
    
    else:
        st.info("👆 Upload some images to get started!")
    
    # Help section
    with st.expander("❓ How does this work?"):
        st.write("""
        **What this app does:**
        1. **Detects Documents**: Tries to find the edges of papers/documents in your photos
        2. **Checks Size**: Makes sure the detected document is big enough (at least 10% of image)
        3. **Fixes Perspective**: Straightens tilted or angled documents (if suitable document detected)
        4. **Cleans Images**: Removes noise and makes text clearer using Sauvola thresholding
        5. **Creates PDF**: Combines all processed images into one PDF file
        
        **Fallback Logic:**
        - If no document edges are found → uses original image
        - If document is too small → uses original image  
        - If perspective correction fails → uses original image
        - If cleaning fails → uses basic grayscale conversion
        
        **Tips for best results:**
        - Take photos with good lighting
        - Try to get the whole document in the frame
        - Make sure the document takes up a good portion of the photo
        - Don't worry if document edges aren't detected - the app will still clean up your image
        - Use the reorder buttons to arrange pages in the right order
        
        **Technical details:**
        - Uses OpenCV for image processing
        - Sauvola thresholding for better text contrast
        - Multiple fallback mechanisms ensure the app never crashes
        - Minimum contour area check prevents processing tiny irrelevant shapes
        """)

# Run the app
if __name__ == "__main__":
    main()