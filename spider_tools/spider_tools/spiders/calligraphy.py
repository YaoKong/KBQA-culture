#!/usr/bin/python
# -*- coding:utf-8 -*-
import re

import pandas as pd
import scrapy

from ..items import CharacterItem
from ..items import CalligraphyItem

class CalligraphySpider(scrapy.Spider):
    """词典网书法Spider"""

    name = "calligraphySpider"
    allowed_domains = ["cidianwang.com"]
    start_urls = [
        "https://www.cidianwang.com/shufazuopin/zhuanti/"
    ]
    custom_settings = {
        'ITEM_PIPELINES': {"spider_tools.pipelines.CalligraphyPipeline": 350},
        'FEEDS': {
            'calligraphy/calligrapher.csv': {
                'format': 'csv',
                'encoding': 'utf8',
                'store_empty': False,
                'item_classes': [CharacterItem],
                'fields': ["name", "alias", "identity", "dynasty", "masterpiece", "region", "summary"],
            },
            'calligraphy/calligraphy.csv': {
                'format': 'csv',
                'encoding': 'utf8',
                'store_empty': False,
                'item_classes': [CalligraphyItem],
                'fields': ["name", "style", "calligrapher",  "url", "carrier", "dynasty", "content"],
            },
        },
    }
    def parse(self, response):
        urls = response.xpath('//div[@class="fr"]/div/a/@href').getall()
        for url in urls:
            yield scrapy.Request(response.urljoin(url), callback=self.parse_chara)


    def parse_chara(self, response):
        chara_item = CharacterItem()
        chara_item["summary"] = ''.join(response.xpath("//div[@class='navmenu']/p/text()").getall())

        chara_item["name"] = re.match(".+(?=书法大全)", response.xpath("string(//h2)").get()).group()
        chara_item["masterpiece"] = list(set(re.findall("[《](.*?)[》]", chara_item["summary"])))
        yield chara_item

        urls = response.xpath("//div[@class='ztlist']/li/div/a/@href").getall()
        for url in urls:
            yield scrapy.Request(response.urljoin(url), callback=self.parse_calligraphy)
    def parse_calligraphy(self, response):
        item = CalligraphyItem()
        col = response.xpath("//*[@id='left']/div[2]/p/text()").getall()
        value = response.xpath("//*[@id='left']/div[2]/p/b/text()").getall()
        col = re.sub("\s|：", "", "#".join(col)).split("#")  # 去除空白符
        info_dict = dict(zip(col, value))
        item["calligrapher"] = info_dict["作者"]
        item["style"] = info_dict["书体"]

        name_xpath = response.xpath("string(//h1)").get()
        if re.search("《.*》", name_xpath) is not None:
            item["name"] = re.search("《.*》", name_xpath).group()
        elif re.search("(?<=·).+", name_xpath) is not None:
            item["name"] = re.search("(?<=·).+", name_xpath).group()

        item["content"] = ''.join(response.xpath("string(//div[@id='left']/div[4])").getall())
        if item["content"].find("碑") != -1 or item["content"].find("刻") != -1:
            item["carrier"] = "碑刻"
        else:
            item["carrier"] = "字帖"

        # [\s|\S]*释文：
        # (?<=(释文：\s*)).+
        item["content"] = re.sub(r"[\s|\S]*(释文：|【释文】)", "", "".join(item["content"]))
        item["content"] = re.sub(r'(\d\s)+.+', '', item["content"])
        item["url"] = response.xpath("//div[@class='left']/div[4]/p//@src").getall()
        yield item

        # 获得下一页的url(为简化没有爬取所有图片)
        # urls = ""