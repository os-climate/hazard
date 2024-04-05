from typing import Dict

from pystac.item_collection import ItemCollection
from pystac_client import Client


def search_stac_items(catalog_url: str, search_params: Dict) -> ItemCollection:
    client = Client.open(catalog_url)
    search = client.search(**search_params)
    items = search.item_collection()
    return items
