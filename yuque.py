# download all images and replace all image urls
# usage: 
#   - python yuque.py --file "xxx.md"

import argparse
import os
import re
import requests

parser = argparse.ArgumentParser()

parser.add_argument("--file", type=str)

args = parser.parse_args()

filename = args.file

if not os.path.exists(filename) or not filename.endswith(".md"):
    print("{} does not exist or not end with .md".format(filename))
    exit(1)

prefix = filename.replace(".md", "")
output_dir = prefix + "-out"
filename_out = os.path.join(output_dir, filename)
image_dir = prefix.replace(" ", "")
abso_image_dir = os.path.join(output_dir, image_dir)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

if not os.path.exists(abso_image_dir):
    os.mkdir(abso_image_dir)

if os.path.exists(filename_out):
    os.remove(filename_out)
    print("out md file={} exists, delete it", filename_out)

print("Output md is: {} image_dir is {}".format(filename_out, abso_image_dir))

with open(filename, 'r', encoding='UTF-8') as f:
    content = f.read()

image_tags = re.findall('(?:!\[.*?\]\(.*?\))', content)

for image_tag in image_tags:
    image_urls = re.findall('(?:!\[(.*?)\]\((.*?)\))', image_tag)[0]
    assert len(image_urls) == 2, image_urls
    image_name, url = image_urls
    image_out_path = os.path.join(abso_image_dir, image_name)
    if not os.path.exists(image_out_path):
        print("Downloading {} from {}".format(image_name, url))
        response = requests.get(url)
        content = content.replace(image_tag, "![{}]({})".format(image_name, os.path.join(image_dir, image_name)))
        with open(image_out_path, "wb") as f:
            f.write(response.content)
    else:
        print("Skip {} because it exists".format(image_name))

with open(filename_out, 'w', encoding='UTF-8') as fout:
    fout.write(content)
