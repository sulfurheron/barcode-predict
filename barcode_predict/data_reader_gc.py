from google.cloud import bigquery
from dotdict import dotdict
from PIL import Image
import os
import numpy as np
import boto3
import datetime
import asyncio
import aiohttp
from urllib.parse import urlparse
import shutil
import pickle
import cv2
import argparse

class ImageDownloader:
    """Downloads scanner images and barcode outlines from S3 and GC.
    
    Iteratively downloads and processes an `_interval_length` hours worth
    of scanner images until `im_total` images are collected. Starting from
    a specified start date, keeps moving the time interval to pull from
    going backward in time.

    Args:
        bucket: either 'kin-sms-media-live' (production) or
            'kin-sms-media-staging' (staging).
        datadir: a string path to a directory where the data will be stored
        interval_length: an integer number of hours representing duration
            of a time interval for a single data batch
        im_total: an integer total number of images to download
        img_dim: a tuple of image dimensions in which to store the images
    """
    def __init__(self,
                 bucket,
                 datadir,
                 interval_length=1,  # in hours
                 im_total=1000000,
                 img_dim=(256, 256)):
        self._interval_length = interval_length
        self.datadir = datadir
        self.img_dim = img_dim
        self.datadir_temp = os.path.join(datadir, "temp")
        self.im_total = im_total
        self._s3 = boto3.client('s3')
        self._bucket = bucket
        self.gc_client = bigquery.Client(project='kin-reporting')

    def query_metadata(self, interval):
        """Builds and send a query to GC to get scanner images metadata.
        
        Args:
            interval: a tuple of string dates representing time interval
                from which to pull the data.
        
        Returns:
            urls: a list of dictionary metadata objects downloaded from GC.
        """
        query = """
            select
              phase_group_id,
              bs.meta_machine_name,
              bs.outline,
              bs.meta_orb_site,
              bs.barcode,
              bs.scanner_id,     
              um.url
            from {barcode_table} bs inner join 
            {media_table} um using (phase_group_id) 
            where bs.meta_event_time >= '{start_time}'
              and bs.meta_event_time <= '{end_time}'  
              and bs.scan_phase = 'scan'      
              and bs.scanner_id in ('1', '2')
              and barcode like 'A%'
              and bs.meta_orb_site in (
                'gapglt',
                'gapfrs',
                'gapfkl',
                'gapcmh'
                 )
              and um.name like 'barcode-scan-%'
              and um.meta_event_time >= '{start_time}'
              and um.meta_event_time <= '{end_time}'   
            limit 1000
        """.format(
            start_time=interval[0],
            end_time=interval[1],
            barcode_table="kin-sort-metrics.raw_metrics.barcode_scans",
            media_table="kin-sort-metrics.raw_metrics.uploaded_media"
        )
        query_job = self.gc_client.query(query)  # Make an API request.
        urls = []
        for row in query_job:
            row = dotdict(row)
            urls.append(row)
        print("Found {} urls".format(len(urls)))
        return urls    

    def update_interval(self, int_end_str):
        """Shifts a time interval by `self._interval_length` hours.
        
        Args:
            int_end_str: a string date reporesenting the end of a current
                time interval.
        
        Returns:
            interval: a tuple of string dates representing a new interval.
            int_end_str: a string date - the end of a new interval.
        """
        date_fmt = "%Y-%m-%d %H:%M:%S"
        delta = datetime.timedelta(hours=self._interval_length)
        int_end_dt = datetime.datetime.strptime(int_end_str, date_fmt)
        int_start_dt = int_end_dt - delta
        int_start_str = int_start_dt.strftime(date_fmt)
        interval = (int_start_str, int_end_str)
        int_end_str = int_start_str
        return interval, int_end_str

    def download_batch(self, int_end_str, max_batch_size=100):
        """Download next batch of images and barcode outlines.
        
        Args:
            int_end_str: a string date - the end of a current time interval.
            max_batch_size: the size of a batch to attempt to download
                at once.
        
        Returns:
            interval: a tuple of string dates representing updated
                time interval.
            int_end_str: a string date - the end of a new interval.
            urls: a list of dictionary metadata objects corresponding to
                downloaded images.
        """
        interval, int_end_str = self.update_interval(int_end_str)
        print("interval", interval)
        urls = self.query_metadata(interval)
        if not len(urls):
            return interval, int_end_str, urls
        # check if the temp directory exists:
        if os.path.isdir(self.datadir_temp):
            try:
                shutil.rmtree(self.datadir_temp)
            except OSError as e:
                print("Error cleaning directory: %s : %s" % (
                    self.datadir_temp,
                    e.strerror
                ))
        os.mkdir(self.datadir_temp)
        for i in range(0, len(urls), max_batch_size):
            self.download_images(
                to_downloads=urls[i:i+max_batch_size],
                out_dir=self.datadir_temp,
            )
        return interval, int_end_str, urls

    def resize_outline(self, outline, orig_shape, new_shape):
        """Resize barcode outline coordinates based on the new image size."""
        new_outline = [
            {
                "x": point["x"] * new_shape[1]/(orig_shape[1] + 0.0),
                "y": point["y"] * new_shape[0]/(orig_shape[0] + 0.0)
            } for point in outline
        ]
        return new_outline

    def resize_and_save(self, filename, current_dir, url, new_shape):
        """Resizes an image and stores it into a new directory structure.
        
        Stores at most 1000 images per directory to ensure efficient random
        file access by OS during training.
        
        Args:
            filename: the name of a file to which the image is currently stored.
            current_dir: a new directory to which to store the image.
            url: a dictionary metadata object corresponding to the image.
            new_shape: a tuple representing image size to resize the image to.
            
        Returns:
            orig_shape: a tuple original image size
            small_outline: a resized barcode outline according to the new_shape
            image_path: a new path to the image file
        """
        try:
            im_path = os.path.join(
                self.datadir_temp,
                url['phase_group_id'],
                filename
            )
            img = np.array(Image.open(im_path))
        except Exception as e:
            print("Exception when reading image file:", e)
            return None, None, None
        orig_shape = img.shape
        small_img = cv2.resize(img, new_shape, cv2.INTER_AREA)        
        small_outline = self.resize_outline(url["outline"], orig_shape, 
                                            new_shape)        
        image_path = os.path.join(current_dir, os.path.splitext(filename)[0])
        image_path += ".npy"
        np.save(image_path, small_img)       
        return orig_shape, small_outline, image_path
    
    def download_images(self, to_downloads, out_dir):
        """Downloads images from S3 based on list of urls."""
        missing_grasp_ids = []
        for to_download in to_downloads:
            path_component = urlparse(to_download['url']).path
            if not os.path.isfile(os.path.join(out_dir, path_component)):
                missing_grasp_ids.append(to_download)
        print(f'Skipping {len(to_downloads) - len(missing_grasp_ids)} '
                          ' grasps (already downloaded).')

        loop = asyncio.get_event_loop()
        n_downloaded = loop.run_until_complete(
            self.download_images_async(
                missing_grasp_ids, out_dir, loop
            )
        )
        return n_downloaded

    async def download_images_async(
        self, to_downloads, out_dir, loop):
        coros = [self._download_images_coro(to_download, out_dir)
                 for to_download in to_downloads]
        await asyncio.gather(*coros)

    async def _download_images_coro(self, to_download, root_dir):
        url = to_download['url']
        image_filename = urlparse(url).path[1:]
        image_filepath = os.path.join(root_dir, image_filename)

        if os.path.isfile(image_filepath):
            return

        os.makedirs(os.path.dirname(image_filepath), exist_ok=True)
        try:
            presigned_url: str = self._s3.generate_presigned_url(
                'get_object',
                Params={
                'Bucket': self._bucket,
                'Key': image_filename,
                }
            )
            async with aiohttp.ClientSession() as session:
                async with session.get(presigned_url) as response:
                    data = await response.read()
                    with open(image_filepath, 'wb') as f:
                        f.write(data)
        except aiohttp.ClientError as ce:
            print('Encountered ClientError when downloading item',
                  {
                      'bucket': self._bucket, 'key': image_filename,
                      'download_location': image_filepath,
                      'exception': ce.__str__(),
                  })

    def sort_files(self, data_dict, urls, interval):
        """Iterates over pulled urls and processes corresponding image files."""
        #plt.figure(figsize=(20, 12))
        if not len(urls):
            return
        for url in urls:
            temp_path = os.path.join(self.datadir_temp, url['phase_group_id'])
            if os.path.exists(temp_path):
                filenames = os.listdir(temp_path)
                for filename in filenames:
                    self.process_image_file(filename, url, data_dict, interval)
        with open(os.path.join(self.datadir, "metadata.pkl"), "wb") as f:
            pickle.dump(data_dict, f)
        print("total images", self.images_num)

    def process_image_file(self, filename, url, data_dict, interval):
        """Processes a single image file."""
        if not 'scanner-{}'.format(url["scanner_id"]) in filename:
            return
        current_dir = os.path.join(self.datadir, str(self.current_dir_id))
        if not os.path.isdir(current_dir):
            os.mkdir(current_dir)
        new_shape = self.img_dim
        orig_shape, small_outline, image_path = self.resize_and_save(
            filename, current_dir, url, new_shape
        )
        if orig_shape is None:
            return
        data_dict['phase_group_id'].append(url["phase_group_id"])
        data_dict["scanner_id"].append(url["scanner_id"])
        data_dict["orb_site"].append(url["meta_orb_site"])
        data_dict["machine_name"].append(url["meta_machine_name"])
        data_dict["barcode"].append(url["barcode"])
        data_dict["interval"].append(interval)
        data_dict['image_file'].append(image_path)
        data_dict['barcode_outline'].append(small_outline)
        self.current_file_id += 1
        if self.current_file_id > 1000:
            self.current_dir_id += 1
            self.current_file_id = 0
        self.images_num += 1        

    def pull_data(self, end_date_str="2021-05-11 00:00:00"):
        """Downloads, processes and adds to the dataset `im_total` images."""
        self.images_num = 0
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
            interval, end_date_str, urls = self.download_batch(end_date_str)
            self.sort_files(data_dict, urls, interval)
            if self.images_num >= self.im_total:
                break
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Scanner Images Downloader')
    parser.add_argument(
        '--datadir',
        type=str,
        default="/home/dkorenkevych/Data/scanner_images"
    )
    parser.add_argument('--images_total', type=int, default=1000000)
    parser.add_argument('--interval_length', type=int, default=1)
    args = parser.parse_args()
    imdl = ImageDownloader(
        bucket="kin-sms-media-live",
        datadir=args.datadir,
        im_total=args.images_total,
        interval_length=args.interval_length
    )
    imdl.pull_data()
