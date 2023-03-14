# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html
import csv

import scrapy
# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
import re

from .items import CharacterItem


class CharacterPipeline:

    def process_item(self, item, spider):
        if isinstance(item, CharacterItem) is False:
            return item
        if item["identity"] is None or item["identity"].find("词语") != -1:
            return item

        item["identity"] = item["identity"].split("、")
        item["summary"] = re.sub("\[.+]|\n|\xa0", "", item["summary"])  # 去除[]以及空白符
        item["basic_value"] = ''.join(item["basic_value"]).replace("\n\n", "##")
        item["basic_value"] = re.sub("\[.+]|\n|\xa0", "",  item["basic_value"]).split("##")
        item["basic_name"] = '##'.join(item["basic_name"])
        item["basic_name"] = re.sub("\[.+]|\n|\xa0", "",  item["basic_name"]).split("##")

        info_dict = dict(zip(item["basic_name"], item["basic_value"]))

        if info_dict.get("本名") is not None:
            item["name"] = info_dict["本名"]
        if info_dict.get("中文名") is not None:
            item["name"] = info_dict["中文名"]

        item["alias"] = []
        if info_dict.get("别名") is not None:
            item["alias"].extend(info_dict["别名"].split("，"))
        if info_dict.get("号") is not None:
            item["alias"].extend(info_dict["号"].split("，"))


        if info_dict.get("时代") is not None:
            item["dynasty"] = info_dict["时代"]
        if info_dict.get("所处时代") is not None:
            item["dynasty"] = info_dict["所处时代"]
        if info_dict.get("出生日期") is not None:
            item["dynasty"] = info_dict["出生日期"]

        if info_dict.get("主要作品") is not None:
            item["masterpiece"] = info_dict["主要作品"]
        if info_dict.get("代表作品") is not None:
            item["masterpiece"] = info_dict["代表作品"]

        if info_dict.get("出生地") is not None:
            item["region"] = info_dict["出生地"]
        if info_dict.get("籍贯") is not None:
            item["region"] = info_dict["籍贯"]

        return item

class CalligraphyPipeline:
    def process_item(self, item, spider):
        print(item)
        return item
class LocationPipeline:
    def process_item(self, item, spider):
        print(item)
        return item