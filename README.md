# KBQA-culture
图数据库：neo4j




代办:

1.本体构建

~~2.poetry数据集的读取，并生成实体、关系~~

3.爬虫爬取相关数据

4.知识融合，设计实体对齐算法

5.实现可视化

6.实现问答系统


## step1 生成数据
运行"build_graph.py"
## step2 导入数据

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