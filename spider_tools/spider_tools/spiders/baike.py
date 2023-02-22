import scrapy

from spider_tools.spider_tools.items import SpiderToolsItem


class BaikeSpider(scrapy.Spider):
    name = "baike"
    allowed_domains = ["baike.baidu.com"]
    start_urls = ["https://baike.baidu.com/item/%E6%9D%8E%E7%99%BD/1043?fromModule=lemma_search-box"]

    def parse(self, response):
        items = SpiderToolsItem()
        summary = response.xpath("//div[@class=`lemma-summary`]")
        info = response.xpath("//div[@data-pid=`card`]")




