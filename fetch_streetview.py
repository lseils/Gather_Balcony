import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
if not API_KEY:
    raise RuntimeError("Google API key not found. Set GOOGLE_MAPS_API_KEY first.")


def get_pano_id(api_key, lat, lng):
    url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    params = {"location": f"{lat},{lng}", "key": api_key}
    r = requests.get(url, params=params)
    data = r.json()
    if data.get("status") == "OK":
        return data["pano_id"], data["location"]["lat"], data["location"]["lng"]
    return None, lat, lng


def download_streetview(api_key, lat, lng, heading, save_folder, file_name, pitch=20, fov=90):
    base_url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        "size": "640x640",
        "location": f"{lat},{lng}",
        "heading": heading,
        "pitch": pitch,
        "fov": fov,
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

    path_coordinates = [
        (33.76483, -84.38206),
        (33.76485, -84.38206),
        (33.76487, -84.38206),
        (33.76489, -84.38206),
        (33.76491, -84.38206),
        (33.76493, -84.38206),
        (33.76495, -84.38206),
        (33.76497, -84.38206),
        (33.76499, -84.38206),
        (33.76501, -84.38206),
        (33.76503, -84.38206),
        (33.76505, -84.38206),
    ]

    facade_headings = [260, 270, 280]
    OPTIMAL_PITCH = 20
    OPTIMAL_FOV = 90

    print("Downloading facade-optimized views...")
    seen_panos = set()
    image_index = 0

    for lat, lng in path_coordinates:
        pano_id, real_lat, real_lng = get_pano_id(API_KEY, lat, lng)
        if pano_id is None:
            print(f"No panorama found at ({lat}, {lng}), skipping.")
            continue
        if pano_id in seen_panos:
            print(f"Skipping duplicate panorama at ({lat}, {lng})")
            continue
        seen_panos.add(pano_id)
        for heading in facade_headings:
            download_streetview(
                api_key=API_KEY,
                lat=real_lat,
                lng=real_lng,
                heading=heading,
                save_folder=OUTPUT_FOLDER,
                file_name=f"facade_{image_index:03d}.jpg",
                pitch=OPTIMAL_PITCH,
                fov=OPTIMAL_FOV
            )
            image_index += 1

    print(f"Done. Downloaded {image_index} images from {len(seen_panos)} unique panoramas.")