from scrapy import cmdline

# 控制台主界面
main = int(input("请输入爬取网站：（1为百度百科人物 2为词典网书法 3为博雅旅游景点）"))
if main == 1:
    cmdline.execute('scrapy crawl baikeSpider'.split())
elif main == 2:
    cmdline.execute('scrapy crawl calligraphySpider'.split())
elif main == 3:
    cmdline.execute('scrapy crawl locationSpider'.split())
else:
    print("输入错误！")
    pass


