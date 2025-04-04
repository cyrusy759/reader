import requests
from bs4 import BeautifulSoup
import random
import pandas as pd
import time

list_url = "https://careers.7-eleven.com/search-jobs"

response = requests.get(list_url)
list_data = response.text
list_soup = BeautifulSoup(list_data, "html.parser")

page_links = list_soup.find_all("a")

links_list = []
job_store = []


for link in page_links:
    href = link.get('href')

    if href.startswith("/job"):
        links_list.append(href)

for job in links_list[:3]:
    job_id = job    
    job_url = f"https://careers.7-eleven.com/{job_id}"
    job_json = {
        "job_title": None,
        "job_description": None,
        "job_lists": None
    }
    response = requests.get(job_url)
    job_data = response.text
    job_soup = BeautifulSoup(job_data, "html.parser")

    job_json["job_title"] = job_soup.find("u").get_text(strip=True)
    job_description = job_soup.find("div", {"class": "description"})
    
    lists_job = job_soup.select("div.description li")
    job_json["job_lists"] = [li.get_text(strip=True) for li in lists_job]

    job_text = job_description.find_all("p")
    job_json["job_description"] = job_text[2].get_text(strip=True)    

    job_store.append(job_json)

    time.sleep(random.uniform(1,3))

print(links_list)


