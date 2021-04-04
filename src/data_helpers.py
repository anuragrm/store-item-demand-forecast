#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
from io import StringIO

import pandas as pd
from azure.storage.blob import BlobServiceClient


def az_load_data(
    blob_dict_inputs,
    az_storage_container_name,
    parse_dates=["date"],
    dict_keys_to_return=["train", "test"],
):
    conn_str = (
        "DefaultEndpointsProtocol=https;"
        f"AccountName={os.getenv('AZURE_STORAGE_ACCOUNT')};"
        f"AccountKey={os.getenv('AZURE_STORAGE_KEY')};"
        f"EndpointSuffix={os.getenv('ENDPOINT_SUFFIX')}"
    )
    blob_service_client = BlobServiceClient.from_connection_string(
        conn_str=conn_str
    )

    df_dict = {}
    for az_blob_name, file_type in blob_dict_inputs.items():
        blob_client = blob_service_client.get_blob_client(
            container=az_storage_container_name, blob=az_blob_name
        )
        blobstring = blob_client.download_blob().content_as_text()
        df_dict[file_type] = pd.read_csv(
            StringIO(blobstring), parse_dates=parse_dates
        )
    return [df_dict[k] for k in dict_keys_to_return]
