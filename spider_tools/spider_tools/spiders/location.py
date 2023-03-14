import scrapy

from ..items import LocationItem


class LocationSpider(scrapy.Spider):
    name = "locationSpider"
    allowed_domains = ["bytravel.cn"]
    # str = "https://baike.baidu.com/item/" + parse.quote("李白")
    start_urls = ["http://www.bytravel.cn/"]

    custom_settings = {
        'ITEM_PIPELINES': {'spider_tools.pipelines.LocationPipeline': 400},
        'FEEDS': {
            'location/location_%(time)s.csv': {
                'format': 'csv',
                'encoding': 'utf8',
                'store_empty': False,
                'item_classes': [LocationItem],
                'fields': ["name", "alias", "province", "city", "urls", "description"],
            },
        },
    }

    def parse(self, response):
        urls = response.xpath('//div[@id="list110"]/a/@href').getall()
        for url in urls:
            yield scrapy.Request(response.urljoin(url), callback=self.parse_more)
    def parse_more(self,response):
        url = response.xpath('//span[@class="listmore"]/a/@href').get()
        yield scrapy.Request(response.urljoin(url), callback=self.parse_city)
    def parse_city(self,response):
        urls = response.xpath('//*[@id="tctitle"]/a/@href').getall()
        for url in urls:
            yield scrapy.Request(response.urljoin(url), callback=self.parse_spot)

        url = response.xpath('//*[@id="list-page"]/ul/li/a')
        if url.xpath("string()").getall()[-1].find("下一页") != -1:
            url = response.urljoin(url.xpath("@href").getall()[-1])
            yield scrapy.Request(url, callback=self.parse_city)
    def parse_spot(self,response):
        item = LocationItem()

        item["name"] = response.xpath("string(//*[@id='page_left']/div[2]/h1)").get()
        item["province"] = response.xpath("string(//*[@id='page_left']/div[1]/div/a[2])").get()
        item["city"] = response.xpath("string(//*[@id='page_left']/div[1]/div/a[3])").get().replace("旅游", "")
        item["urls"] = []
        for url in response.xpath('//div[@align="center"]//img/@src').getall():
            if url.find("http") != -1:
                item["urls"].append(url)
        item["description"] = ''.join(response.xpath("//p/text()").getall())
        yield item