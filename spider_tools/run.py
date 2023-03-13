from scrapy import cmdline

# 控制台主界面
main = int(input("请输入爬取网站：（1为百度百科 2为词典网书法 3为site03）"))
if main == 1:
    cmdline.execute('scrapy crawl baikeSpider'.split())
elif main == 2:
    cmdline.execute('scrapy crawl calligraphySpider'.split())
else:
    print("输入错误！")
    pass

print('-----爬虫启动-----')

