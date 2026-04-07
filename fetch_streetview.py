import os
import requests
from dotenv import load_dotenv

# load enviornment variables from the .env file
load_dotenv()

API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
if not API_KEY:
    raise RuntimeError("Google API key not found. Set GOOGLE_MAPS_API_KEY first.")

def download_streetview(api_key, lat, lng, heading, save_folder, file_name, pitch=20, fov=60):
    """
    Downloads a single Google Street View image.
    Adjust pitch (look up/down) and fov (zoom) to isolate the building facade.
    """
    base_url = "https://maps.googleapis.com/maps/api/streetview"
    
    # Parameters for the API
    params = {
        "size": "640x640",         # Max free tier resolution (width x height)
        "location": f"{lat},{lng}",
        "heading": heading,        
        "pitch": pitch,            # OVERCOMING CARS: 20 tilts the camera up over street clutter
        "fov": fov,                # ZOOM: 60 acts like a telephoto lens, flattening perspective
        "key": api_key,
        "return_error_code": "true"
    }
    
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        os.makedirs(save_folder, exist_ok=True)
        file_path = os.path.join(save_folder, file_name)
        
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded: {file_name} (Heading: {heading}, Pitch: {pitch}, FOV: {fov})")
    else:
        print(f"Failed to download {file_name}. (Status: {response.status_code})")

if __name__ == "__main__":
    OUTPUT_FOLDER = "street_images"

    # ==========================================
    # FACADE-CAPTURE SEQUENCE (WEST-FACING)
    # ==========================================

    path_coordinates =[
        (33.76433, -84.38209),
        (33.76443, -84.38209),
        (33.76453, -84.38209),
        (33.76463, -84.38209),
    ]

    # Headings centered on WEST (270) with oblique angles
    facade_headings =[230, 245, 260, 275, 295]

    # Best settings to avoid street clutter:
    # Pitch ~15 to 25 looks up at the 2nd/3rd story
    # FOV ~50 to 60 zooms past the sidewalk
    OPTIMAL_PITCH = 20 
    OPTIMAL_FOV = 60

    image_index = 0

    print("Downloading facade-optimized views...")
    for lat, lng in path_coordinates:
        for heading in facade_headings:
            download_streetview(
                api_key=API_KEY,
                lat=lat,
                lng=lng,
                heading=heading,
                save_folder=OUTPUT_FOLDER,
                file_name=f"facade_{image_index:03d}.jpg",
                pitch=OPTIMAL_PITCH,
                fov=OPTIMAL_FOV
            )
            image_index += 1
            
            