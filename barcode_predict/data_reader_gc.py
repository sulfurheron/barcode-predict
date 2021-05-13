from google.cloud import bigquery
from dotdict import dotdict
from PIL import Image, ImageDraw
from io import BytesIO
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import boto3
import datetime
import asyncio
import aiohttp
from urllib.parse import urlparse
import shutil
from threading import Thread
import glob
import pickle
import cv2

class ImageDownloader:

    def __init__(self,
                 bucket,
                 datadir,
                 datadir_temp,
                 interval_length=1):
        self._interval_length = interval_length
        self.datadir = datadir
        self.datadir_temp = datadir_temp
        self._s3 = boto3.client('s3')
        self._bucket = bucket
        self.gc_client = bigquery.Client(project='kin-reporting')

    def query_pg_ids(self, interval):
        query = """
            select
              phase_group_id,
              bs.meta_machine_name,
              bs.outline,
              bs.meta_orb_site,
              bs.barcode,
              bs.scanner_id,     
              um.url
            from kin-sort-metrics.raw_metrics.barcode_scans bs inner join 
            kin-sort-metrics.raw_metrics.uploaded_media um using (phase_group_id) 
            where bs.meta_event_time >= '{start_time}'
              and bs.meta_event_time <= '{end_time}'  
              and bs.scan_phase = 'scan'      
              and (bs.scanner_id = '1' OR bs.scanner_id = '2')
              and barcode like 'A%'
              and (bs.meta_orb_site = 'gapglt' OR bs.meta_orb_site = 'gapfrs' OR bs.meta_orb_site = 'gapfkl' OR bs.meta_orb_site = 'gapcmh')
              and um.name like 'barcode-scan-%'
              and um.meta_event_time >= '{start_time}'
              and um.meta_event_time <= '{end_time}'   
            limit 1000
        """.format(
            start_time=interval[0],
            end_time=interval[1]
        )
        query_job = self.gc_client.query(query)  # Make an API request.
        urls = []
        for row in query_job:
            # Row values can be accessed by field name or index.
            row = dotdict(row)
            urls.append(row)
        print("Found {} urls".format(len(urls)))
        return urls

    def download_images(self, to_downloads, out_dir):
        missing_grasp_ids = []
        for to_download in to_downloads:
            path_component = urlparse(to_download['url']).path
            if not os.path.isfile(os.path.join(out_dir, path_component)):
                missing_grasp_ids.append(to_download)
        print(f'Skipping {len(to_downloads) - len(missing_grasp_ids)} grasps '
                          '(already downloaded).')

        loop = asyncio.get_event_loop()
        n_downloaded = loop.run_until_complete(self.download_images_async(
            missing_grasp_ids, out_dir, loop))
        return n_downloaded

    async def download_images_async(
        self, to_downloads, out_dir, loop):
        coros = [self._download_images_coro(to_download, out_dir) for to_download in to_downloads]
        await asyncio.gather(*coros)

    async def _download_images_coro(self, to_download, root_dir):
        url = to_download['url']
        image_filename = urlparse(url).path[1:]
        image_filepath = os.path.join(root_dir, image_filename)

        if os.path.isfile(image_filepath):
            return

        os.makedirs(os.path.dirname(image_filepath), exist_ok=True)
        try:
            presigned_url: str = self._s3.generate_presigned_url('get_object', Params={
                'Bucket': self._bucket,
                'Key': image_filename,
            })
            async with aiohttp.ClientSession() as session:
                async with session.get(presigned_url) as response:
                    data = await response.read()
                    with open(image_filepath, 'wb') as f:
                        f.write(data)
        except aiohttp.ClientError as ce:
            print('Encountered ClientError when downloading item', extra={
                'bucket': self._bucket, 'key': image_filename,
                'download_location': image_filepath, 'exception': ce.__str__(),
            })

    def download(self, to_downloads, max_batch_size=100):
        for i in range(0, len(to_downloads), max_batch_size):
            self.download_images(
                to_downloads=to_downloads[i:i+max_batch_size],
                out_dir=self.datadir_temp,
            )

    def update_interval(self, int_end_str):
        int_end_dt = datetime.datetime.strptime(int_end_str, "%Y-%m-%d %H:%M:%S")
        int_start_dt = int_end_dt - datetime.timedelta(hours=self._interval_length)
        int_start_str = int_start_dt.strftime("%Y-%m-%d %H:%M:%S")
        interval = (int_start_str, int_end_str)
        int_end_str = int_start_str
        return interval, int_end_str

    def download_batch(self, int_end_str):
        interval, int_end_str = self.update_interval(int_end_str)
        print("interval", interval)
        urls = self.query_pg_ids(interval)
        if not len(urls):
            return interval, int_end_str, urls
        # check the directory exists:
        if os.path.isdir(self.datadir_temp):
            try:
                shutil.rmtree(self.datadir_temp)
            except OSError as e:
                print("Error cleaning directory: %s : %s" % (self.datadir_temp, e.strerror))
        os.mkdir(self.datadir_temp)
        self.download(urls)
        return interval, int_end_str, urls

    def resize_outline(self, outline, orig_shape, new_shape):
        new_outline = [
            {
                "x": point["x"] * new_shape[1]/(orig_shape[1] + 0.0),
                "y": point["y"] * new_shape[0]/(orig_shape[0] + 0.0)
            } for point in outline
        ]
        return new_outline

    def im_resize_and_save(self, filename, current_dir, url, new_shape):
        try:
            img = np.array(Image.open(os.path.join(self.datadir_temp, url['phase_group_id'], filename)))
        except Exception as e:
            print("Exception when reading image file:", e)
            return None, None, None
        orig_shape = img.shape
        small_img = cv2.resize(img, new_shape, cv2.INTER_AREA)

        #pil_img = Image.fromarray(small_img)
        small_outline = self.resize_outline(url["outline"], orig_shape, new_shape)
        # pil_mask = Image.new('L', new_shape, 0)
        # outline = [(point["x"], point["y"]) for point in small_outline]
        # ImageDraw.Draw(pil_mask).polygon(outline, outline=1, fill=1)
        #mask = np.array(mask)
        # img_bytes = BytesIO()
        # pil_img.save(img_bytes, 'png')
        # mask_bytes = BytesIO()
        # pil_mask.save(mask_bytes, 'png')
        # jpgs = [img_bytes]
        image_path = os.path.join(current_dir, os.path.splitext(filename)[0]) + ".npy"
        np.save(image_path, small_img)
        #pil_img.save(image_path, 'png')
        # with open(image_path, "wb") as f:
        #     pickle.dump(jpgs, f)
        #pil_img.save(os.path.join(current_dir, filename), 'jpeg')
        return orig_shape, small_outline, image_path

    def sort_files(self, data_dict, urls, interval):
        #plt.figure(figsize=(20, 12))
        if not len(urls):
            return
        for url in urls:
            if os.path.exists(os.path.join(self.datadir_temp, url['phase_group_id'])):
                filenames = os.listdir(os.path.join(self.datadir_temp, url['phase_group_id']))
                for filename in filenames:
                    if 'scanner-{}'.format(url["scanner_id"]) in filename:
                        current_dir = os.path.join(self.datadir, str(self.current_dir_id))
                        if not os.path.isdir(current_dir):
                            os.mkdir(current_dir)
                        new_shape = (256, 256)
                        orig_shape, small_outline, image_path = self.im_resize_and_save(filename, current_dir,
                                                                                        url, new_shape)
                        if orig_shape is None:
                            continue
                        data_dict['phase_group_id'].append(url["phase_group_id"])
                        data_dict["scanner_id"].append(url["scanner_id"])
                        data_dict["orb_site"].append(url["meta_orb_site"])
                        data_dict["machine_name"].append(url["meta_machine_name"])
                        data_dict["barcode"].append(url["barcode"])
                        data_dict["interval"].append(interval)
                        data_dict['image_file'].append(image_path)
                        data_dict['barcode_outline'].append(small_outline)
                        #shutil.copy(os.path.join(self.datadir_temp, url['phase_group_id'], filename), current_dir)
                        self.current_file_id += 1
                        if self.current_file_id > 1000:
                            self.current_dir_id += 1
                            self.current_file_id = 0
                        self.images_num += 1

                        # if self.images_num < 9:
                        #     img = np.array(Image.open(os.path.join(self.datadir_temp, url['phase_group_id'], filename)))
                        #     orig_shape = img.shape
                        #     new_shape = (256, 256)
                        #     print("shape", img.shape)
                        #     small_img = cv2.resize(img, new_shape, cv2.INTER_AREA)
                        #     mask = Image.new('L', new_shape, 0)
                        #     outline = [(point["x"], point["y"])  for point in small_outline]
                        #     ImageDraw.Draw(mask).polygon(outline, outline=1, fill=1)
                        #     mask = np.array(mask)
                        #     print("mask vals", np.max(mask), np.min(mask))
                        #     print("outline", url["outline"])
                        #     ax = plt.subplot(4, 4, 2 * (self.images_num - 1) + 1)
                        #     plt.imshow(small_img, cmap='gray')
                        #     plt.subplot(4, 4, 2 * self.images_num)
                        #     plt.imshow(mask, cmap='gray')
                        #     #outline = [(point["x"] * new_shape[1]/(orig_shape[1] + 0.0), point["y"] * new_shape[0]/(orig_shape[0] + 0.0)) for point in url["outline"]]
                        #     ax.add_patch(patches.Polygon(np.array(outline), color='g', fill=False))
                        # else:
                        #     plt.show()
        with open(os.path.join(self.datadir, "metadata.pkl"), "wb") as f:
            pickle.dump(data_dict, f)
        print("total images", self.images_num)

    def pull_data_batch(self):
        self.images_num = 0
        int_end_str = '2021-05-11 00:00:00'
        self.current_dir_id = 0
        self.current_file_id = 0
        data_dict = {
            "phase_group_id": [],
            "scanner_id": [],
            "image_file": [],
            "barcode_outline": [],
            "orb_site": [],
            "machine_name": [],
            "barcode": [],
            "interval": [],
        }
        while True:
            interval, int_end_str, urls = self.download_batch(int_end_str)
            self.sort_files(data_dict, urls, interval)
            if self.images_num > 1000000:
                break


if __name__ == "__main__":
    imd = ImageDownloader(
        bucket="kin-sms-media-live",
        datadir="/home/dkorenkevych/Data/scanner_images",
        datadir_temp="/home/dkorenkevych/Data/scanner_images/temp")
    imd.pull_data_batch()
    # urls = imd.query_pg_ids()
    # imd.download(urls)
    # plt.figure(figsize=(20, 12))
    # count = 0
    # for url in urls:
    #     filenames = os.listdir(os.path.join("images", url['phase_group_id']))
    #     print("filenames", filenames)
    #     for file in filenames:
    #         if "scanner-1" in file:
    #             count += 1
    #             if count < 17:
    #                 img = np.array(Image.open(os.path.join("images", url['phase_group_id'], file)))
    #                 print("shape", img.shape)
    #                 print("outline", url["outline"])
    #                 ax = plt.subplot(4, 4, count)
    #                 plt.imshow(img, cmap='gray')
    #                 outline = [(point["x"], point["y"]) for point in url["outline"]]
    #                 ax.add_patch(patches.Polygon(np.array(outline), color='g', fill=False))
    # plt.show()