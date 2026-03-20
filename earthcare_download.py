from pystac_client import Client
from dotenv import load_dotenv
import earthcarekit as eck
from earthcarekit.utils.time ._day_night import get_day_night_mask
import numpy as np

import os
import requests
import tqdm


bbox = [-30, -30, 30, 30]
datetime = ['2025-09-01T00:00:00Z',
            '2025-10-01T23:59:59Z']
product = "AC__TC__2B"
var = "synergetic_target_classification"


load_dotenv()


# not user specific
CLIENT_ID = "offline-token"
CLIENT_SECRET = "p1eL7uonXs6MDxtGbgKdPVRAmnGxHpVE"

MAAP_TOKEN = os.getenv("MAAP_TOKEN")
# Print the first 10 characters for verification
print(f"Using MAAP token: {MAAP_TOKEN[:10]}...")


catalog_url = 'https://catalog.maap.eo.esa.int/catalogue/'
catalog = Client.open(catalog_url)

# Select one or more collection(s)
EC_COLLECTION = ['EarthCAREL2Validated_MAAP']
EC_multiple_COLLECTIONS = [
    'EarthCAREL2Validated_MAAP', 'EarthCAREL1Validated_MAAP']

search = catalog.search(
    collections=EC_COLLECTION,
    # For example filter by product type and baseline. Use boolean logic for multi-filter queries
    filter="productType = 'AC__TC__2B' and productVersion = 'ba'",
    bbox=bbox,
    datetime=datetime,
    method='GET',  # This is necessary
    max_items=1000
    # max_items=5  # Adjust as needed, given the large amount of products it is recommended to set a limit if especially if you display results in pandas dataframe or similiar
)


items = list(search.items())  # Get all items as a list
# items = search.item_collection() # Get all items as a STAC ItemCollection
results = search.matched()
print(f"Number of items found: {results}")


# filter items by day/night
coordinates = [item.geometry['coordinates'] for item in items]
# do the centroid of the coordinates, centroid in lat/lon not quite correct but should be good enough for day/night mask
items_center_lon = [np.mean([coord[0] for coord in item.geometry['coordinates']]) for item in items]
items_center_lat = [np.mean([coord[1] for coord in item.geometry['coordinates']]) for item in items]
items_center_time = [item.datetime.replace(tzinfo=None) if hasattr(item.datetime, 'replace') else item.datetime for item in items]


day_mask = get_day_night_mask(items_center_time, items_center_lat, items_center_lon) 
items_day = [item for item, is_day in zip(items, day_mask) if is_day]
print(f"Number of items found: {len(items_day)}")


url = "https://iam.maap.eo.esa.int/realms/esa-maap/protocol/openid-connect/token"
data = {
    "client_id": CLIENT_ID,
    "client_secret": CLIENT_SECRET,
    "grant_type": "refresh_token",
    "refresh_token": MAAP_TOKEN,
    "scope": "offline_access openid"
}

response = requests.post(url, data=data)
response.raise_for_status()

response_json = response.json()
access_token = response_json.get('access_token')
print("Access token retrieved successfully.")

if not access_token:
    raise RuntimeError("Failed to retrieve access token from IAM response")


# 2. Download the H5 file from an item
for item in tqdm.tqdm(items_day):
    h5_url = item.assets["enclosure_h5"].href
    filename = os.path.join(
        "./DATA/EC", item.assets["enclosure_h5"].extra_fields["file:local_path"])

    response = requests.get(
        h5_url,
        headers={"Authorization": f"Bearer {access_token}"},
        stream=True
    )
    with open(filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded {filename}")
