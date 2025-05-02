from PIL import Image, ImageChops, ImageEnhance
import os

def perform_ela(image: Image.Image, quality=90) -> Image.Image:
    # Save the image at a lower quality
    temp_filename = "temp_ela.jpg"
    image.save(temp_filename, 'JPEG', quality=quality)
    
    # Reload the compressed image
    compressed_image = Image.open(temp_filename)

    # Get the difference between original and compressed
    ela_image = ImageChops.difference(image, compressed_image)

    # Enhance the difference
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0 / max_diff if max_diff != 0 else 1
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    # Clean up temp file
    os.remove(temp_filename)

    return ela_image
