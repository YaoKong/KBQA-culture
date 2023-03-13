# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class CharacterItem(scrapy.Item):
    name = scrapy.Field()
    summary = scrapy.Field()
    basic_name = scrapy.Field()
    basic_value = scrapy.Field()
    masterpiece = scrapy.Field()
    alias = scrapy.Field()
    dynasty = scrapy.Field()
    region = scrapy.Field()
    identity = scrapy.Field()

class CalligraphyItem(scrapy.Item):
    style = scrapy.Field()
    calligrapher = scrapy.Field()
    content = scrapy.Field()
    name = scrapy.Field()
    url = scrapy.Field()
    carrier = scrapy.Field()  #书法形式，如碑刻，字帖(拓本)
    dynasty = scrapy.Field()

class LocationItem(scrapy.Item):
    name = scrapy.Field()
    alias = scrapy.Field()
    province = scrapy.Field()
    city = scrapy.Field()
    description = scrapy.Field()
