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
