# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class SpiderToolsItem(scrapy.Item):
    name = scrapy.Field()
    description = scrapy.Field()
    masterpiece = scrapy.Field()
    alias = scrapy.Field()
    dynasty = scrapy.Field()
    region = scrapy.Field()
    identity = scrapy.Field()

