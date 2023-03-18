import re

import pandas as pd
import scrapy

from ..items import LocDictItem
from urllib import parse
import re
class locDictSpider(scrapy.Spider):
    name = "locDict"
    allowed_domains = ["baike.baidu.com"]

    custom_settings = {
        'FEEDS': {
            'locDict/locDict_%(time)s.json': {
                'format': 'jsonlines',
                'encoding': 'utf8',
                'store_empty': False,
            },
        },
    }

    def start_requests(self):
        with open("./cityName.txt", "r", encoding='utf-8') as f:
            for line in f:
                new_url = "https://baike.baidu.com/item/" + re.sub("\n", "", line)
                yield scrapy.Request(
                    url=new_url,
                    callback=self.parse
                )

    def parse(self, response):

        basic_value = response.xpath("//dd[@class='basicInfo-item value']").xpath("string(.)").getall()
        basic_name = response.xpath("//dt[@class='basicInfo-item name']").xpath("string(.)").getall()

        basic_value = ''.join(basic_value).replace("\n\n", "##")
        basic_value = re.sub("\[.+]|\s", "",  basic_value).split("##")
        basic_name = '##'.join(basic_name)
        basic_name = re.sub("\[.+]|\s", "",  basic_name).split("##")

        info = dict(zip(basic_name, basic_value))

        item = None
        if info.get("中文名") is None:
            url = response.xpath("//div[@class='para']/a[@target='_blank']/@href").xpath("string(.)").get()
            yield scrapy.Request(response.urljoin(url), callback=self.parse)
        else:
            item = LocDictItem()
            item["name"] = info["中文名"]
            if info.get("别名") is not None:
                item["alias"] = info["别名"].split("、")

        yield item




