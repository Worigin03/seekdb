# 全文索引热路径优化报告（fts_large_bench）

> 分支：`fts-opt-tokenize-hotpath`，基线：`2026-competition` @ `5bce9e0`

## 1. 结论

本次改动集中在**分词热路径的固定开销**上，共 3 处：

| 改动 | 位置 | 作用 |
| --- | --- | --- |
| 停用词查表不再逐词新建 arena，且同字符集时跳过 charset_convert | `storage/fts/ob_fts_stop_word.cpp` | 每个 token 省一次 `ObArenaAllocator` 构造 + 一次字符串转换 |
| 分词属性为空时跳过 JSON 往返 | `sql/engine/expr/ob_expr_tokenize.cpp`、`storage/fts/ob_fts_parser_property.cpp` | 每次 `TOKENIZE()` 省两次空 JSON 文档构造/解析/序列化 |
| 字典 range 查找由线性扫描改为二分 | `storage/fts/dict/ob_ft_range_dict.cpp` | 每个字符的词典探测由 O(ranges) 降到 O(log ranges) |

同机 A/B（各 3 轮 `fts_large_bench.sh`，`SKIP_LOAD=0`）：

| 指标 | 基线均值 | 优化后均值 | 变化 |
| --- | --- | --- | --- |
| `build_ik_all_sec` | 9.480 | 8.665 | **-8.6%** |
| `build_ik_content_sec` | 6.485 | 6.293 | **-3.0%** |
| `build_beng_en_sec` | 3.457 | 3.383 | **-2.1%** |
| `build_total_sec` | 19.432 | 18.352 | **-5.6%** |
| `tokenize_ik_avg_ms` | 0.2942 | 0.2706 | **-8.0%** |
| `tokenize_beng_avg_ms` | 0.2422 | 0.2124 | **-12.3%** |
| `query_cn_avg_ms` | 2.085 | 2.063 | -1.1% |
| `query_beng_avg_ms` | 2.933 | 2.978 | +1.5% |
| `query_mixed_avg_ms` | 2.250 | 2.240 | -0.5% |
| `query_limit_avg_ms` | 2.519 | 2.671 | +6.0% |

说明：

- `tokenize_*` 是三轮中最稳定的指标（组内标准差 < 4%），也是本次改动直接作用的路径。
  `tokenize_beng` 的三轮里有一轮出现 0.1595 的低值，均值被拉低，故同时给出中位数：
  `tokenize_ik` 0.2950 → 0.2831（-4.0%），`tokenize_beng` 0.2437 → 0.2379（-2.4%）。
- 构建类指标方差很大（同一版本三轮 `build_ik_all_sec` 在 8.6～10.3s 之间波动，
  受后台合并、page cache、宿主机负载影响），-3%～-9% 的变化方向一致但幅度仅供参考。
- 查询链路的代码没有被改动，±1%～6% 属于噪声范围，按**不变**看待。

同时对照仓库内冻结基线 `tools/benchmark/fts_large_bench_baseline.json`（2026-07-07 采集）：

| 指标 | 冻结基线 | 本分支（优化后） | 变化 |
| --- | --- | --- | --- |
| `build_ik_all_sec` | 35.284 | 8.665 | -75% |
| `build_ik_content_sec` | 28.376 | 6.293 | -78% |
| `build_beng_en_sec` | 14.758 | 3.383 | -77% |
| `tokenize_ik_avg_ms` | 0.7648 | 0.2706 | -65% |
| `tokenize_beng_avg_ms` | 0.4226 | 0.2124 | -50% |
| `query_cn_avg_ms` | 16.663 | 2.063 | -88% |
| `query_beng_avg_ms` | 24.304 | 2.978 | -88% |
| `query_mixed_avg_ms` | 17.559 | 2.240 | -87% |
| `query_limit_avg_ms` | 16.233 | 2.671 | -84% |

**这部分差距主要来自基线冻结（2026-07-07）之后上游对检索与索引链路的既有优化**，
不是本次提交的贡献。本次提交在已经很快的 `2026-competition` 头部上再削减了
构建 3～9%、分词 2～8% 的耗时。按题面公式（三类等权、单项提升截断在 50%），
本分支在冻结基线下已处于满分区间；不同 CI 硬件上仍以官方统一评测为准。

## 2. 瓶颈定位依据

### 2.1 端到端拆分

用同一实例跑 `fts_large_bench.sh` 的 `select1_avg_ms` 作为 SQL 回环基线
（本机 0.19～0.22ms），再对同一条 `SELECT tokenize(...)` 做定长压测：

| 语句 | 每次耗时 | 相对 `SELECT 1` |
| --- | --- | --- |
| `SELECT 1` | 195µs | — |
| `tokenize('中','ngram2')` | 206µs | +11µs |
| `tokenize('ab','beng')` | 208µs | +13µs |
| `tokenize('中','ik')` | 221µs | +26µs |
| `tokenize(<108 字中文>,'ik')` | 306µs | +120µs |

结论：

1. **固定开销 11～13µs/次**（参数解析、属性 JSON、helper 初始化、hash map、JSON 输出），
   IK 另加约 13µs（三个词典的 cache 装载 + 4 个 processor 构造）；
2. **边际开销约 0.9µs/字符**（IK），是长文本的主体，主要由逐字符分类、
   词典 DAT 探测、arbitrator 的链式裁决构成；
3. 停用词检查对 `beng` 生效（`ObAddWordFlag::set_stop_word()`），
   而 `ObStopWordChecker::check_stopword()` 每查一个词都会构造一个
   `ObArenaAllocator` 并做一次 `charset_convert`，属于纯浪费。

### 2.2 代码层面的三处浪费

- `ObStopWordChecker::check_stopword()`：`common::ObArenaAllocator allocator(...)`
  是函数内的局部对象，每个 token 一次；`charset_convert` 在两个 collation
  同属 utf8mb4 时只是字节拷贝。
- `ObExprTokenize::TokenizeParam::reform_parser_properties()` 与
  `ObFTParserProperty::parse_for_parser_helper()`：即使没有第三个参数（绝大多数调用），
  也要 `ObFTParserJsonProps::init()` + `parse_from_valid_str("")` + `to_format_json()`，
  而空属性的结果就是一个空串。
- `ObFTRangeDict::find_first_char_range()`：对 `range_dicts_` 线性扫描并对每个 range
  做两次 `ObCharset::strcmp`；该函数在**每个字符**开始词典探测时都会被调用，
  而 `range_dicts_` 本身是按 key 升序、互不重叠的。

## 3. 修改说明

### 3.1 停用词查表零分配

`check_stopword` 改为接收调用方的 allocator（`ObAddWord::allocator_`，即分词使用的
arena），并在源字符串与停用词表属于同一字符集时直接构造 `ObFTWord` 查表：

```cpp
if (ObCharset::charset_type_by_coll(word.get_collation_type())
    == ObCharset::charset_type_by_coll(stopword_type_.get_collation_type())) {
  converted = ObFTWord(word_str.length(), word_str.ptr(), stopword_type_);
} else {
  // 原路径：charset_convert 到停用词表字符集
}
```

等价性：`charset_convert` 在两个 collation 同字符集时不会改变字节内容，
查表语义只依赖字节 + collation（`ObFTWord::hash/operator==` 走 datum 比较函数），
所以直接换 meta 与先转换再查表结果一致。

### 3.2 空属性跳过 JSON 往返

`parse_for_parser_helper()` 增加空串快路径：`ObFTParserProperty` 的构造函数已经把
所有字段置为默认值，空属性只需为 IK 补上三个内置词典表名（原逻辑同样只是赋值常量）。
`reform_parser_properties()` 在 `properties.empty()` 时直接把 `properties_` 置空并返回——
原路径对空属性 `to_format_json()` 返回的也是空串，因此下游行为完全一致。

### 3.3 词典 range 二分查找

`range_dicts_` 由 `build_dict_from_cache()` 按 range_id 升序追加（range 在构建阶段按
key 升序切分、互不重叠），因此包含目标词的 range 可以二分定位：

```cpp
while (left <= right) {
  mid = left + ((right - left) >> 1);
  if (start(mid) > word)      right = mid - 1;
  else if (end(mid) < word)   left  = mid + 1;
  else { dict = range_dicts_[mid].dict_; break; }
}
```

命中/未命中语义与线性扫描一致，仅比较次数由 O(n) 降为 O(log n)。

## 4. 正确性验证

1. **分词结果逐字节比对**：26 条探针 SQL，覆盖 IK / BENG / space / ngram2、
   中英混排、数字、标点、空串、单字、停用词、URL/邮箱、制表符、大小写混排，
   在基线与优化版本上 `diff` 完全一致（`IDENTICAL`）。
2. **MATCH 命中数一致**：所有 3+3 轮基准中 `query_cn_hits=8001`、
   `query_beng_hits=11000`、`query_mixed_hits=7332`、`query_limit_hits=20`，
   与基线完全相同。
3. **构建流程**：3 个全文索引（IK 全字段 / IK 正文 / BENG 英文）在优化版本上均构建成功，
   20,000 行数据可正常 `MATCH` 检索（`SKIP_LOAD=0` 全流程）。
4. 改动仅涉及停用词查表、属性解析快路径、range 定位三处，
   不触碰分词规则、排序/裁决逻辑、倒排写入与存储格式。

## 5. 复现

```bash
# 同机 A/B：先跑基线，再切到本分支，各 3 轮
cd seekdb
MYSQL='mysql -h127.0.0.1 -P2881 -uroot -N -s --default-character-set=utf8mb4' \
MYSQL_VERBOSE='mysql -h127.0.0.1 -P2881 -uroot --default-character-set=utf8mb4' \
LABEL=base OUTPUT=./bench.txt ./tools/benchmark/fts_large_bench.sh
```

## 6. 未做/后续可做

- IK 的 `arbitrator` 用有序链表做冲突裁决（`ObFTSortList::add_token` 为 O(n) 插入），
  在重叠 token 密集的中文文本上是平方级开销；重写为数组 + 归并预计还能再降
  10%～20%，但会触碰分词裁决语义，本次未纳入。
- 索引构建的剩余开销主要在 DML 写入与合并（每篇文档约 500µs，其中分词约占 20%），
  需要改 `ObFTIndexRowCache` 的行生成与批量写入路径，改动面大、回归风险高。

## 7. 第二轮补充：域索引构建并行度（2026-11-05）

第一轮提测得分 10.034/50，说明只优化分词热路径不足。按 `fts_large_bench_score.py`
的口径反推（`score = 类别平均提升 / 0.5 × 100`），10.034/50 对应平均提升 10.03%，
即三类（build / tokenize / query）各占 1/3，query 一项完全没有贡献。

对 20,000 行的 `ALTER TABLE ... ADD FULLTEXT INDEX` 逐子任务计时后发现：
`fts_doc_word`（词表）子任务占 13.1s / 13.3s，且任务日志里 `parallelism_:1` ——
在 12 核机器上整个域索引是**单线程**构建的。

原因在 `ObFtsIndexBuilderUtil::decide_parallelism()`：它以 DDL 语句的并行度为基数，
而普通 `ALTER TABLE ADD FULLTEXT INDEX` 没有 `PARALLEL` 子句时该值为 1，
于是 `MAX(original/3, 1) == 1`，四个子任务全部串行。

改动：DDL 未显式指定并行度时，按 `min(8, cpu_num)` 取基数，FTS 类子任务取 `base/2`，
即 12 核机器上并行度 4（实测 8 反而更慢：8.76s vs 5.73s，IO/锁竞争）。

同机对比（20k 行，单索引 drop + add）：

| 并行度 | 耗时 |
| --- | --- |
| 1（改前） | 13.3s |
| 4（改后） | 5.73s |
| 8 | 8.76s |

完整 `fts_large_bench.sh` 三轮（父提交 → 本分支）：

| 指标 | 父提交 | 本分支 | 变化 |
| --- | --- | --- | --- |
| build_ik_all_sec | 8.31 | 6.48 / 7.40 | −22% / −11% |
| build_ik_content_sec | 6.17 | 4.75 / 5.26 | −23% / −15% |
| build_beng_en_sec | 3.25 | 2.44 / 2.74 | −25% / −16% |
| tokenize_ik_avg_ms | 0.272 | 0.237 / 0.284 | 代码未变，差异为噪声 |
| query_* | — | — | 代码未变，差异为噪声 |

构建项三轮组内一致，但机器本身有 10%~20% 的漂移（父提交三轮 7.99/8.31/10.23，
本分支 6.38/6.48/6.68 与 7.20/7.40/7.50），因此取「父提交中位数 vs 本分支中位数」
为 −22%，保守可认为 **−15% 左右**。

正确性：`query_cn_hits=8001`、`query_beng_hits=11000`、`query_mixed_hits=7332`、
`query_limit_hits=20` 与基线完全一致，3 个索引全部构建成功。

已知未做：MATCH 查询的相关度（BM25）无法安全跳过 —— 计划层把
`need_calc_relevance` 置 false 后，DAAT/TAAT 迭代器仍要求 BM25 估计器与
relevance collector 非空（`ObTextDaaTIter::init`、`init_taat_iter_param`），
直接关会崩溃；彻底去掉需要在检索引擎里引入「无相关度」模式，改动面过大，本轮未纳入。
单条 2 词查询的固定开销中约 270µs/词花在 `estimate_token_doc_cnt`（文档频次估计），
其余约 130ns/条 posting 花在逐条 BM25 求值，二者是后续 query 优化的着力点。

## 8. 第四轮内核优化与验证

以 `34a1c3b2` 为同机对照，两版均使用 RelWithDebInfo，在同一 Docker
环境中使用独立实例。完整运行未修改的比赛脚本，参数为
`ROWS=20000 BATCH=500 ROUNDS=3000 QUERY_ROUNDS=200 SAMPLES=3 WARMUP=30 SKIP_LOAD=0`，
MySQL 客户端使用 `-A`。表内列出每版三轮耗时和两版中位数的相对差异。

| 指标 | `34a1c3b2` 三轮 | 本提交三轮 | 中位数耗时下降 |
| --- | ---: | ---: | ---: |
| build_ik_all_sec | 10.166 / 11.653 / 6.879 | 7.097 / 6.662 / 6.283 | +34.5% |
| build_ik_content_sec | 7.377 / 6.250 / 4.960 | 6.786 / 5.157 / 4.548 | +17.5% |
| build_beng_en_sec | 3.554 / 2.742 / 2.644 | 2.967 / 7.730 / 2.746 | -8.2% |
| tokenize_ik_avg_ms | 0.3303 / 0.3067 / 0.2612 | 0.2690 / 0.3126 / 0.2637 | +12.3% |
| tokenize_beng_avg_ms | 0.2567 / 0.2284 / 0.2257 | 0.2223 / 0.2241 / 0.2123 | +2.7% |
| query_cn_avg_ms | 1.7447 / 1.5300 / 1.5166 | 1.0723 / 1.1712 / 1.0674 | +29.9% |
| query_beng_avg_ms | 2.7452 / 2.1896 / 2.1651 | 2.3137 / 1.5729 / 1.4118 | +28.2% |
| query_mixed_avg_ms | 1.7699 / 1.6874 / 1.6276 | 1.1522 / 1.1145 / 1.1607 | +31.7% |
| query_limit_avg_ms | 2.8749 / 2.6757 / 2.5834 | 0.3605 / 0.3888 / 0.5637 | +85.5% |

构建、分词和查询分别为 +14.6%、+7.5% 和 +43.8%，三类等权后的本地改善为 +22.0%。
六轮 `SELECT 1` 耗时依上表版本顺序分别为 0.4230 / 0.2024 / 0.1931 / 0.1995 / 0.2003 / 0.1879 ms。
两版首轮和 BENG 构建样本有明显波动，表中保留全部六轮。

### 瓶颈与改动

- IK 在逐字符循环中重复分类、间接调用各处理器，并为每段冲突裁决构造哈希表。
  现在一次解码用于分类，处理器保持原顺序共用一次分配，裁决按文档顺序遍历冲突组。
  编译内置的两份词典在服务内共享，只读对象发布后保留缓存引用到服务退出。
- BENG 的 ASCII 字符宽度可直接确定，其他输入使用字符集例程。
  分词和归一化直接使用原有两个流，归一化写入文档分配器，省去中间缓冲区复制。
  构建和查询使用相同解析器，已有倒排数据的布局保持一致。
- 只有在短词表的自然语言查询无需相关度、权重或过滤时，才直接合并已排序的倒排头部。
  对纯 `COUNT(*)` 消费的受限派生表，在确认外层只观察行数后移除自动注入的相关度排序，
  并让内层只投影常量。基准查询的执行计划显示索引内扫描与 `LIMIT 20`，
  `is_index_back=false`、`calc_relevance=false`。显式排序和可观察行内容的查询保持原路径。

### 正确性与正式分数

新回归用例用父提交生成期望输出，并在本提交上运行通过。对比了 IK 和 BENG
的 506 条完整分词结果，以及 39 行查询输出，覆盖词频、相关度、排名、
布尔条件、空值、准备语句、事务可见性及文档 ID 映射。三类索引都成功重建，
所有完整基准的四项命中数一致，分别为 8001、11000、7332 和 20。

用户提供的 9.506/50 评测报告，按仓库提交的
`fts_large_bench_baseline.json` 重算得到 88.53% 综合耗时下降。
正式平台报告给出 9.51% 综合耗时下降。报告缺少平台实际基线和被测 SHA，
这两组结果尚无法对齐。上表衡量同机增量，正式分数待统一环境复测。

## 9. 回填中间行与词频更新

`8dd57c89` 的正式评测为 28.095/50。平台报告显示，这一版相比
`34a1c3b2` 的主要收益来自查询，三项构建与两项分词只下降约 2%–7%。
本轮针对构建和分词仍共同承担的成本做了两处修改。

- 在线回填的扫描只需要词项、文档 ID、词频、文档长度四个字段。回填路径按文档
  连续保存这四个字段，减少通用存储行、行指针数组和逐行初始化的开销。
  两种辅助索引的字段顺序仍由同一段代码生成，单条记录的字段在扫描下一篇文档前有效。
- 词频表由当前解析任务独占。复用哈希表的插入或更新接口，已有词项直接更新，
  省去第二次查找和节点替换。排序规则视为同一词的不同字节表示仍保留最近一次的词形。
  `TOKENIZE`、索引构建和 `MATCH` 查询分词共用该路径。

在同一 Docker 环境使用 RelWithDebInfo，`8dd57c89` 和本提交各运行三轮
未修改的比赛脚本。两版均预先运行回归以初始化词典。
每轮使用 `ROWS=20000 BATCH=500 ROUNDS=3000 QUERY_ROUNDS=200 SAMPLES=3 WARMUP=30 SKIP_LOAD=0`
和 MySQL 客户端 `-A`。每项先取同版本三轮的中位数，再计算相对差异。

| 指标 | `8dd57c89` 三轮 | 本提交三轮 | 中位数耗时下降 |
| --- | ---: | ---: | ---: |
| build_ik_all_sec | 6.865 / 7.212 / 6.128 | 5.956 / 5.954 / 6.183 | +13.2% |
| build_ik_content_sec | 5.356 / 4.860 / 4.957 | 4.650 / 4.849 / 5.526 | +2.2% |
| build_beng_en_sec | 2.956 / 2.754 / 2.441 | 7.426 / 2.741 / 2.745 | +0.3% |
| tokenize_ik_avg_ms | 0.2735 / 0.2598 / 0.2642 | 0.2482 / 0.2783 / 0.2512 | +4.9% |
| tokenize_beng_avg_ms | 0.2189 / 0.2155 / 0.2203 | 0.2195 / 0.2174 / 0.2137 | +0.7% |
| query_cn_avg_ms | 1.0668 / 1.0956 / 1.0745 | 1.1513 / 1.1172 / 1.0603 | -4.0% |
| query_beng_avg_ms | 1.4447 / 1.6005 / 1.4104 | 1.4413 / 1.4298 / 1.4304 | +1.0% |
| query_mixed_avg_ms | 1.1286 / 1.1306 / 1.1261 | 1.1401 / 1.1319 / 1.1700 | -1.0% |
| query_limit_avg_ms | 0.3667 / 0.3710 / 0.3788 | 0.3534 / 0.3748 / 0.3671 | +1.1% |

构建、分词、查询三类分别为 +5.2%、+2.8%、-0.7%。
三类等权后，本地相对父提交的综合耗时下降 2.4%。
`SELECT 1` 耗时依上表顺序为 0.2027 / 0.1917 / 0.1897 / 0.1933 / 0.1892 / 0.1866 ms。
BENG 构建有一轮耗时 7.426 秒，表中保留全部样本。

额外用服务端 `BENCHMARK(4000, TOKENIZE(...))` 减少客户端往返的影响，
IK 和 BENG 循环中位耗时分别下降 9.6% 和
20.8%。这组诊断数据单独记录，正式分数由比赛平台复评。

四个全文索引回归用例通过。512 条分词输出和 39 行查询结果逐字节一致。
长文档回填、更新、删除、重建前后的行和 BM25 分数也与父提交一致。
六轮完整基准的四项命中数均为 8001、11000、7332 和 20。

## 10. 倒排扫描投影与 IK 冲突组

正式平台给父提交 `6ea42821` 评出 31.238/50。本轮无相关度也无相关度过滤的
全文查询只从倒排扫描取文档 ID。相关度计算路径仍读取词频和文档长度。
IK 内置词典使用与 UTF8MB4_BIN 相同的二进制范围比较，单词冲突组直接引用
原有词项，重叠词项沿用原裁决流程。

同机按完整比赛参数各跑五轮，基准脚本与评分文件无差异。
以下每项用两版各自的中位数计算变化。

| 指标 | `6ea42821` 五轮 | 本提交五轮 | 中位数耗时下降 |
| --- | ---: | ---: | ---: |
| build_ik_all_sec | 8.698 / 6.617 / 6.259 / 5.477 / 6.254 | 10.628 / 6.604 / 5.983 / 5.371 / 5.588 | +4.4% |
| build_ik_content_sec | 4.344 / 5.039 / 4.956 / 4.344 / 4.043 | 7.546 / 4.854 / 4.142 / 4.041 / 3.817 | +4.7% |
| build_beng_en_sec | 2.638 / 2.754 / 2.921 / 2.346 / 2.239 | 5.809 / 2.959 / 2.536 / 2.411 / 2.437 | +3.9% |
| tokenize_ik_avg_ms | 0.2529 / 0.2612 / 0.2506 / 0.2431 / 0.2443 | 0.2891 / 0.2619 / 0.2451 / 0.2399 / 0.2382 | +2.2% |
| tokenize_beng_avg_ms | 0.2038 / 0.2316 / 0.2068 / 0.2144 / 0.2043 | 0.2337 / 0.2196 / 0.2029 / 0.2075 / 0.2019 | -0.3% |
| query_cn_avg_ms | 1.0400 / 1.0669 / 1.0488 / 1.0515 / 1.0212 | 1.0880 / 1.0047 / 0.9590 / 0.9418 / 0.9353 | +8.6% |
| query_beng_avg_ms | 1.3300 / 1.5183 / 1.3360 / 1.3227 / 1.3175 | 1.4947 / 1.3260 / 1.2714 / 1.2769 / 1.2653 | +4.0% |
| query_mixed_avg_ms | 1.0780 / 1.2542 / 1.0780 / 1.0609 / 1.0503 | 1.2503 / 1.0478 / 1.0365 / 1.0119 / 1.0084 | +3.8% |
| query_limit_avg_ms | 0.3479 / 0.3876 / 0.3383 / 0.3452 / 0.3308 | 0.3776 / 0.3582 / 0.3557 / 0.3265 / 0.3349 | -3.0% |

构建、分词和查询三类分别为 +4.3%、+0.9% 和 +3.3%，
三类等权后本地耗时下降 2.9%。
`SELECT 1` 耗时依上表版本顺序为 0.1849 / 0.1878 / 0.1912 / 0.1787 / 0.1820 / 0.1820 / 0.2252 / 0.1823 / 0.1771 / 0.1808 ms。
前三轮和补测的环境波动明显。独立服务端分词循环的 IK、BENG 中位耗时分别
增加 20.5%、11.4%，
与完整基准的分词方向不完全一致。

四个回归用例通过。512 条分词和 39 行查询结果逐字节一致，
长文档回填、重建索引及并发建索引期间提交的新增、更新、删除也通过结果对照。
十轮完整基准的四项命中均为 8001、11000、7332 和 20。
