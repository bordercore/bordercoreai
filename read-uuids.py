import csv

import django

django.setup()

from blob.models import Blob

blobs = Blob.objects.filter(content__isnull=False).order_by("?")

with open("uuids.txt", "w", newline="") as file:
    writer = csv.writer(file)
    data = []
    for blob in blobs:
        print(blob.name)
        # file.write(str(blob.uuid) + "\n")
        data.append([str(blob.uuid), blob.name])
        # writer.writerows([str(blob.uuid) + "\n", blob.name])
    writer.writerows(data)
    # print(data)
# with open("uuids.txt", "w") as file:
#     for blob in blobs:
#         print(blob.name)
#         file.write(str(blob.uuid) + "\n")
