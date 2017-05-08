## 机器学习实战
> 代码中用到了[ND4J](http://nd4j.org/),ND4J是JVM的科学计算库，并为生产环境设计，亦即例程运行速度快，RAM要求低。
[google guava](https://github.com/google/guava),Guava工程包含了若干被Google的 Java项目广泛依赖 的核心库，
例如：集合 [collections] 、缓存 [caching] 、原生类型支持 [primitives support] 、并发库 [concurrency libraries] 、
通用注解 [common annotations] 、字符串处理 [string processing] 、I/O 等等

- ###  K-近邻算法
    > 存在一个样本数据集合（训练样本集合），并且样本集中的每个数据都存在一个标签（每一个数据与所述类别的关系）
    输入没有标签的数据之后，将新数据的每个特征与样本数据集合的数据进行比较，然后算法提取样本数据集合中最相似
    （最近邻）数据的标签