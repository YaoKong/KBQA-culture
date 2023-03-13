import re

import pandas as pd
import scrapy

from ..items import CharacterItem


from urllib import parse

class BaikeSpider(scrapy.Spider):
    name = "baikeSpider"
    allowed_domains = ["baike.baidu.com"]
    # str = "https://baike.baidu.com/item/" + parse.quote("李白")
    # start_urls = [str]

    custom_settings = {
        'ITEM_PIPELINES': {'spider_tools.pipelines.CharacterPipeline': 300},
        'FEEDS': {
            '%(name)s/%(name)s_%(time)s.csv': {
                'format': 'csv',
                'encoding': 'utf8',
                'store_empty': False,
                'fields': ["name", "alias", "identity", "dynasty", "masterpiece", "region", "summary"],
            },
        },
    }

    def start_requests(self):
        chara_list = pd.read_csv("characterList.csv")["作者"].tolist()
        count = 0
        for chara in chara_list:
            count += 1
            if count % 10 == 0:
                print("第{}条url".format(count))
            new_url = "https://baike.baidu.com/item/" + parse.quote(chara)
            yield scrapy.Request(
                url=new_url,
                callback=self.parse
            )

    def parse(self, response):
        item = CharacterItem()
        item["identity"] = response.xpath("//div[@class='lemma-desc']").xpath("string(.)").get()
        item["summary"] = response.xpath("string(//div[@label-module='lemmaSummary'])").get()
        item["basic_value"] = response.xpath("//dd[@class='basicInfo-item value']").xpath("string(.)").getall()
        item["basic_name"] = response.xpath("//dt[@class='basicInfo-item name']").xpath("string(.)").getall()
        return item




