import scrapy

from spider_tools.spider_tools.items import LocationItem


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
                'fields': ["name", "alias", "province", "city", "description"],
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
    def parse_city(self,response):
        urls = response.xpath('//*[@id="tctitle"]/a/@href').getall()
        for url in urls:
            yield scrapy.Request(response.urljoin(url), callback=self.parse_spot)