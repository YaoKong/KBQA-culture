# KBQA-culture
图数据库：neo4j

TO DO:

1.本体构建

~~2.poetry数据集的读取~~

~~3.爬虫爬取相关数据~~

4.知识抽取，生成实体和关系

4.知识融合，设计实体对齐算法

5.实现可视化

6.实现问答系统


## step1 生成数据
### 1.读取数据集
运行"build_graph.py"

```3090跑完大概要40分钟```

或

直接下载已经生成好的数据
链接：https://pan.baidu.com/s/1YHY4mDVePS1xtith6M0Diw 

提取码：atri

### 2.爬虫爬取

``` 已爬好的文件会在后续给出```

运行spider_tools文件夹下的run.py，按提示爬取四个网站，生成四个文件夹，请让
每个文件夹仅保留一个csv文件。

## step2 知识抽取
### 1.关系抽取
TO DO

## step3 知识融合
TO DO
## step4 知识存储

### 离线导入:
1.复制data下的csv文件到neo4j项目的import文件夹

  Neo4j Destop可直接点击项目的“Open”->"Open folder"->"Import"快速定位
  
2.关闭neo4j项目

3.打开neo4j项目终端，执行以下命令
```angular2html
bin\neo4j-admin database import full 
--nodes import\character.csv 
--nodes import\poetry.csv 
--nodes import\location.csv 
--relationships import\indite_rels.csv 
--relationships import\refer_rels.csv
--trim-strings=true dbName
```

注意!输入命令时可选参数之间只有空格没有换行，Linux需用/分隔路径

4.启动neo4j项目，并创建名为dbName的数据库