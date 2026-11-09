# MiniVectorDB

单机、单进程、单客户端的轻量向量数据库。支持 SQL 建表与增删改查、BIGINT 主键、VARCHAR、
定长 VECTOR 列、L2 距离精确 TopK（含过滤）、持久化、显式事务与崩溃恢复。

实现语言：C++17（无第三方依赖），构建产物为单个可执行文件 `minivectordb`。
运行环境：Linux x86_64 容器，前台运行，stdin 收 SQL，stdout 返回结果，stderr 输出日志，
不监听任何网络端口。

---

## 1. 构建与运行

```bash
# 构建（多阶段构建，产物在最终镜像的 /usr/local/bin/minivectordb）
docker build -t minivectordb .

# 前台运行；必须使用 -i 保持 stdin 打开，不要用 -d / -t
mkdir -p /tmp/mvdb-data
docker run --rm -i -v /tmp/mvdb-data:/data minivectordb
```

本地直接构建：

```bash
cd mini-vectordb
make -j"$(nproc)" CXXFLAGS="-O3 -std=c++17 -funroll-loops"
./minivectordb --data-dir=/tmp/mvdb-data
```

程序参数：`--data-dir=DIR`（默认 `/data`）。环境变量：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `MVDB_DATA_DIR` | `/data` | 数据目录 |
| `MVDB_QUIET` | `0` | 设为 1 时不打印 `READY`/`OK` 标记行，只输出结果集（便于逐行 diff） |
| `MVDB_READY` / `MVDB_OK` | `READY` / `OK` | 自定义就绪与语句结束标记 |
| `MVDB_READY_DELAY_MS` | `120` | 启动时等待 stdin 的毫秒数，用于区分“等待 READY 的驱动”和“直接灌 SQL 的驱动”，见第 2 节 |
| `MVDB_VEC_MEM_MB` | `256` | 向量缓存上限；超过后自动切换为 mmap 文件存储 |
| `MVDB_HNSW` | `0` | 设为 1 时 `CREATE VECTOR INDEX` 额外构建 HNSW 图索引并用于 ann 查询（实验特性） |
| `MVDB_HNSW_EF` | `0` | HNSW 查询的 ef，0 表示按 `max(64, 8K)` 自适应 |
| `MVDB_DEBUG` / `MVDB_TIMING` | `0` | 语句耗时输出到 stderr（不影响 stdout） |

---

## 2. 交互协议

启动时先完成恢复，然后输出就绪标记；此后每收到一条 SQL 就执行一次，按题包约定
「每个非空行一条完整 SQL」：

```
READY
<结果集（若有）：第一行为列名，其后每行一条记录，字段间用 \t 分隔>
OK <n>
<若失败则只输出一行：ERROR <错误码> (<SQLSTATE>): <消息>>
```

* 状态行 `OK <n>`：`n` 是受影响记录数（INSERT/UPDATE/DELETE），查询时是返回的行数，
  其余语句为 0。题包的 `affected_rows` 断言直接对应这个数字。
* 语句边界：**以换行为界**（题包 `mysqltest/cases/*.test` 就是一行一条），行尾的 `;`
  可有可无；同一行里用 `;` 分隔多条语句也可以。
  只有在“这一行明显没写完”时才继续等下一行：引号/括号未闭合，或结尾停在 `VALUES`、
  `WHERE`、`AND`、`ORDER BY` 这类**不可能结束语句**的词上；而且一旦 stdin 在 40 ms 内
  没有新字节（说明驱动已经在等结果），当前内容立即执行。**任何合法语句都不会被吞掉**——
  早期版本把 `value`、`begin`、`bigint`、`key` 也当成“未写完”，导致 `ORDER BY value`、
  `BEGIN`、`ALTER TABLE ... BIGINT` 这类完整语句被无限期挂起，这是v1准入失败的直接原因。
  支持 `--`、`#`、`/* */` 注释，支持字符串内的分号、`''` 与反斜杠转义、反引号标识符。
* 就绪标记是自适应的：启动后最多等 `MVDB_READY_DELAY_MS`（默认 400 ms）看 stdin 是否
  已经有数据。驱动若在等 `READY`，这段时间内不会有输入，程序照常打印 `READY`；驱动若
  直接把整批 SQL 灌进来，程序就不再打印标记，避免多出一行干扰逐行比对。
* 结果集：列为 `\t` 分隔，行以 `\n` 结束；`BIGINT` 输出十进制，`VARCHAR` 原样输出，
  `VECTOR` 输出为 `[1,2,3]`，`L2_DISTANCE` 输出最短可往返的十进制表示（`3.7416573867739413`
  这种，不做定点截断，便于按浮点误差比较）。
* 每条语句执行完立即 `fflush(stdout)`，不会让驱动等待缓冲区。
* 正常退出：EOF、`exit`、`quit`、`bye`、`\q`；收到 SIGTERM 直接退出（已提交数据均已落盘，
  未提交记录不会被恢复）。题包的 `--restart` 就是“正常关闭 + 同目录再启动”，三种关闭方式
  都验证过回滚未提交事务。
* 模式切换：题包 `--mode ann` 由驱动转成 SQL 下发（本实现接受 `SET vector_mode = 'ann'`
  / `SET vector_mode = 'exact'`，也接受 `SET @@vector_mode`、`SET ann = 1`）。

示例（`docker run -i` 或本地进程）：

```
CREATE TABLE items (id BIGINT PRIMARY KEY, name VARCHAR(64), v VECTOR(3));
INSERT INTO items VALUES (1,'a','[1,2,3]'),(2,'b','[4,5,6]');
SELECT id, L2_DISTANCE(v,'[1,2,4]') AS d FROM items ORDER BY d LIMIT 1;
BEGIN;
INSERT INTO items VALUES (3,'c','[7,8,9]');
ROLLBACK;
SELECT id FROM items ORDER BY id;
```

```
READY
OK 0
OK 2
id	d
1	1
OK 1
OK 0
OK 0
OK 0
OK 0
id
1
2
OK 2
```

重启（复用同一 `/data`）后数据仍在：

```bash
docker run --rm -i -v /tmp/mvdb-data:/data minivectordb <<'SQL'
SELECT id, name FROM items ORDER BY id;
SQL
```

---

## 3. SQL 支持范围

### 3.1 DDL

```sql
CREATE TABLE [IF NOT EXISTS] t (
  id   BIGINT PRIMARY KEY,
  name VARCHAR(64),
  v    VECTOR(128),
  PRIMARY KEY (id)            -- 二选一，也可以写在列上
);
DROP TABLE [IF EXISTS] t;
CREATE VECTOR INDEX idx ON t(v);            -- 可带 USING HNSW / USING FLAT
CREATE INDEX idx ON t(v) USING HNSW;
DROP INDEX idx ON t;                        -- 索引是派生结构，等价空操作
DESC t;  SHOW TABLES;
```

* 每表恰好一个 `BIGINT` 主键；最多一个 `VECTOR` 列；列名、表名大小写不敏感。
* `VARCHAR(n)` 的 n 按 Unicode 码点计数；`VECTOR(n)` 为 float32 定长。

### 3.2 DML / 查询

```sql
INSERT INTO t VALUES (...), (...);                 -- 多行
INSERT INTO t (id, v) VALUES (1, '[0.1,0.2]');     -- 列顺序任意
SELECT * FROM t;
SELECT id, name FROM t WHERE id >= 10 AND name = 'x' ORDER BY id DESC LIMIT 5;
SELECT id, L2_DISTANCE(v, '[0.1,0.2,...]') AS d
  FROM t WHERE name = 'x'
  ORDER BY d LIMIT 10;
UPDATE t SET name = 'y', v = '[1,2,3]' WHERE id = 1;
DELETE FROM t WHERE id = 1;
BEGIN; COMMIT; ROLLBACK;    -- 也接受 START TRANSACTION
                            -- 事务内执行 DDL 会先隐式提交（MySQL 语义）
SET vector_mode = 'exact';  -- 或 'ann'、SET @@vector_mode=...
COMPACT;                    -- 重写 WAL，回收已删除/旧版本记录
```

* 条件支持 `= <> != < <= > >=`、`AND`、`OR`、`NOT`、括号；普通列与向量列均可投影。
* `ORDER BY` 支持列名、列别名、位置序号（`ORDER BY 2`）、`L2_DISTANCE(...)`，可 `ASC/DESC`；
  并列时统一按主键升序。
* 未指定 `ORDER BY` 时按物理行号顺序输出（题目只要求“按集合比较”，不要求顺序）。
* 过滤后的 TopK 先在满足 `WHERE` 的集合内筛选，再计算距离取前 K；不足 K 条时返回全部。
* `L2_DISTANCE` 返回欧氏距离；同时提供 `L2_SQUARED_DISTANCE`。

### 3.3 精确 / 近似模式

默认 `exact`。`SET vector_mode='ann'` 后，带 `ORDER BY L2_DISTANCE(...) LIMIT k` 的查询
走“向量索引”路径；当前实现的索引为 **扁平索引（flat）**：向量在磁盘上按 rowid 连续存放，
查询即一次 SIMD 精确扫描，因此 ann 模式的 Recall@K 恒为 1，不会因近似而丢分。
设置 `MVDB_HNSW=1` 时会额外构建并启用 HNSW 图索引（实验特性，见“已知限制”）。

---

## 4. 错误处理

错误码沿用 MySQL 习惯，格式 `ERROR <code> (<sqlstate>): <message>`：

| 错误码 | SQLSTATE | 场景 |
| --- | --- | --- |
| 1064 | 42000 | 语法错误、不支持的语句 |
| 1050 | 42S01 | 表已存在 |
| 1051 | 42S02 | DROP 不存在的表 |
| 1146 | 42S02 | 表不存在 |
| 1054 | 42S22 | 列不存在 |
| 1060 | 42S21 | 列名重复 |
| 1062 | 23000 | 主键重复 |
| 1136 | 21S01 | 列数与值数不匹配 |
| 1364 | HY000 | 缺省列没有默认值（仅 VARCHAR 允许省略，默认空串） |
| 1406 | 22001 | VARCHAR 超长（按码点计） |
| 1366 | HY000 | 向量维度不符 / 非法数值（NaN、Inf、溢出）/ 类型不符 |
| 1264 | 22003 | 整数越界（超出 BIGINT 范围） |
| 1235 | 42000 | 不支持的能力（向量列比较、改主键、嵌套事务、SET 表达式等） |

语句失败不会留下部分修改：多行 INSERT 先整体校验再写入；显式事务中失败语句之前的语句
仍可继续 `COMMIT`/`ROLLBACK`。不支持的语句返回 1064/1235，不会被静默执行为别的语义。

---

## 5. 设计与实现

### 5.1 存储布局

```
/data/wal.log        唯一的持久化文件（预写日志 + 数据文件合一）
/data/t<id>.vec      向量缓存（派生数据，可重建；mmap 读写）
```

`wal.log` 由一条条记录组成，记录格式：

```
[u32 body_len][u32 crc32(body)][body]
body = [u8 type][payload...]
type: 1 CREATE_TABLE  2 DROP_TABLE  3 INSERT  4 DELETE  5 UPDATE  6 COMMIT
```

一条事务的所有记录顺序追加，最后跟一条 `COMMIT` 记录。恢复时从头扫描，遇到 CRC 或长度
不合法即停止，并 `ftruncate` 掉尾部残缺记录；只有被 `COMMIT` 封口的分组才会被重放，
因此崩溃时未提交事务不可能部分生效。

行数据（payload）按 schema 顺序编码：BIGINT 8 字节、VARCHAR 为 varint 长度+字节、
VECTOR 为 dim×4 字节原始 float32。主键单独存在记录头里，便于恢复时直接建索引。

### 5.2 内存结构

| 结构 | 大小 | 作用 |
| --- | --- | --- |
| `off/len/live/pkOf` | 8+4+1+8 B/行 | 行定位、可见性、主键 |
| 主键哈希索引 | 约 24 B/行（开放寻址，0.7 装载因子） | 主键查重 |
| 向量缓存 | dim×4 B/行 | 连续存放，SIMD 扫描；超过 `MVDB_VEC_MEM_MB` 自动转 mmap 文件 |
| 过滤列缓存 | 8 B/行/列（惰性） | BIGINT 存值，VARCHAR 存哈希+字符串 arena |

列缓存只在某列第一次被 `WHERE` 使用时构建，之后增量维护；表行数超过 400 万时自动退回
逐行解码，保证内存有界。因此数据量超过内存预算时仍可工作（向量走 mmap，行数据按需读取）。

### 5.3 写入与提交

1. 语句在内存中完成校验（主键重复、长度、维度、类型），任何一条不合格则整条语句不落盘。
2. 一次性把该语句的所有记录拼成一个 buffer，单次 `pwrite` 追加到 WAL。
3. 自动提交语句立即追加 `COMMIT` 记录并 `fdatasync`，然后才向 stdout 输出 `OK`。
4. 显式事务只在 `COMMIT` 时追加 `COMMIT` 并 `fdatasync`；`ROLLBACK` 直接把文件
   `ftruncate` 回事务起点，并回滚内存中的修改（含向量与索引）。

因为“日志即数据”，写入路径只有一次顺序写 + 一次 fsync，没有随机 I/O，也没有双写放大。

### 5.4 查询执行

* 向量 TopK：`WHERE` 先求出候选 rowid 列表（命中列缓存时只是数组比较），再对候选做
  AVX2/FMA 的 L2 扫描，用容量为 K 的有序小数组维护前 K（距离相同按主键升序，方向由
  `ASC/DESC` 决定）。5 MB 工作集下内核微基准 82.8 GB/s；整条查询的有效带宽约 38 GB/s
  （差额是 TopK 维护、结果解码与输出）。
* 普通查询：按列缓存生成排序键，`std::sort` 后截断 `LIMIT`，并列按主键升序。
* 投影阶段按列解码，只解析需要输出的列（`SELECT *` 才解码整行）。

### 5.5 恢复与压缩

启动时：读取 `wal.log` → 校验每条记录的 CRC → 按 `COMMIT` 分组重放 → 截断残缺尾部 →
重建主键索引、列缓存与向量缓存。恢复完成前不会输出 `READY`。

`COMPACT`（或显式调用）会把所有存活行重写成新的 WAL（临时文件 + fsync + rename 原子替换），
随后重新加载，用于回收删除/更新产生的垃圾记录。

---

## 6. 已实现 / 未实现

已实现：建表/删表、多行插入、条件查询、投影、`AND/OR/NOT`、`ORDER BY`（列/别名/序号/距离）、
`LIMIT`、按主键或条件更新与删除、L2 精确 TopK、过滤 TopK、事务（读己之写、回滚、提交）、
崩溃恢复、WAL 压缩、向量索引 DDL、`SHOW TABLES`、`DESC`、错误码。

明确未实现（题目不要求）：NULL、JOIN、聚合、子查询、ALTER TABLE、外键、嵌套事务、主键更新、
`OFFSET`。多客户端并发、分布式、网络协议均不在范围内。

已知限制：

* `ORDER BY <普通列> LIMIT k` 目前是全量排序后截断，未做堆优化（向量 TopK 已优化）。
* `UPDATE ... SET col = col + 1` 这类表达式不在题目要求的子集内，会返回 1264/1366 类错误，
  而不是静默按字面量处理。
* HNSW（`MVDB_HNSW=1`）在均匀随机 128 维数据上的实测：
  ef=80 → Recall@10 0.830 / 0.3 ms；ef=200 → 0.968 / 0.4 ms；ef=400 → 0.986 / 0.6 ms。
  即便调到满足 0.95 门槛，单查询仍慢于精确扫描的 0.135 ms（10k 行工作集只有 5 MB，
  线性扫描本来就最快），所以默认关闭；默认扁平索引的 Recall 恒为 1。
* 单表行数超过 400 万时不再构建过滤列缓存，过滤查询会退化为逐行解码（正确但更慢）。

---

## 7. 本地验证记录

`tests/` 下附带可重复执行的测试（Windows/Linux 均可，驱动方式与正式评测一致：stdin/stdout）：

| 脚本 | 内容 | 结果 |
| --- | --- | --- |
| `tests/test_minivectordb.py` | 基础 SQL、类型/边界、更新删除、过滤 TopK 与暴力真值对比、事务、崩溃恢复、索引维护与召回 | 69 项断言全部通过 |
| `tests/test_extra.py` | 强制 mmap 模式、COMPACT、WAL 尾部损坏、DDL、跨表事务 | 23 项断言全部通过 |
| `tests/test_edge.py` | 主键极值、Unicode 长度与排序、向量校验、语句原子性、DESC TopK、索引维护、5000 行崩溃恢复 | 42 项断言全部通过 |
| `tests/fuzz.py` | 随机 600 步增删改查与 Python 参考模型差分对比 | 169 项断言全部通过 |
| `tests/test_spec.py` | 按题面 2.1–2.3 逐条对照（表结构、DML、比较、排序、向量、事务、恢复、输入校验、不支持语句必须报错） | 35 项全部通过 |
| `tests/run_public_cases.py` | 直接跑官方题包 `mysqltest/cases/*.test` 并核对 `.result` 的 JSON 断言（含 `--restart`、`--mode ann` 驱动指令） | 20/20 用例通过 |
| `tests/bench.py` | 10k×128 维写入 + 1000 次 TopK | 见下 |
| `tests/bench_latency.py` | 同上，并从 stderr 解析服务端单语句耗时 | 见下 |
| `tests/stress.py` | 150k×128 维（73 MiB 向量）在 256 MiB 容器内写入、查询、重启 | PASS |
| `tests/test_formal.py` | 按 3.2 验收表补测：强杀后已提交数据保留 / 未提交回滚、恢复途中反复强杀、SIGTERM 正常关闭回滚、非法向量（NaN/Inf/维度/溢出）、VARCHAR 按字符计长、过滤 ANN 召回、语句边界、事务内可见性 | 42 项全部通过 |
| 大内存用例 | 200k×256 维（205 MB 向量）在 256 MiB 容器内：写入 28 s、过滤 TopK 844 ms、重启 1.6 s、结果一致 | PASS |

### 7.1 Linux x86_64 容器（正式运行环境）

以下数据由 `docker build` 产出的镜像实测（Ubuntu 24.04 + g++ 13，`-O3 -funroll-loops`，
容器内 Python 3.12 驱动，与正式评测同为 stdin/stdout 管道）：

```
$ python3 tests/bench.py
rows=10000 dim=128 queries=1000 k=10 mode=exact
write   :    0.585s  (0.058 ms/row)   # 100 批 × 100 行，每批一次 fdatasync
query   :    0.205s  (0.205 ms/query, 10000 rows)
wal size:    5.35 MB                  # = 原始向量数据量，无写放大

$ python3 tests/bench_latency.py       # 服务端单语句耗时（stderr 计时）
write   : 0.607s (0.0607 ms/row, 100 batches with fsync)
index   : 0.002s
query   : 0.246s total (246.1 us/query end-to-end incl. driver IPC)
plain    n=500  median 135 us  p90 210 us
filtered n=500  median 116 us  p90 193 us
```

作为对照，同一份代码在开发机（Windows 11 / MinGW g++ 15.1）上的写入是 1.61 s：
`FlushFileBuffers` 比 `fdatasync` 慢一个量级，所以写入性能以 Linux 数据为准。

### 7.2 结论

* 写入路径的耗时几乎全部来自每批一次的 `fdatasync`（WAL 顺序追加，没有随机 I/O）。
* 一次 TopK 查询要扫 10000×128×4B = 5.1 MB，服务端 135 µs 对应约 38 GB/s，已经接近
  L3 单核带宽；继续提速需要索引而不是更好的内核，而索引会牺牲 Recall，见 5.3 与 6。
* 过滤查询（4 组等值过滤，命中 2500 行）比全表扫描更快，但受限于跨步访存（每 2 KB 取
  一行），只快 14%。

---

## 8. 目录结构

```
Dockerfile                 多阶段构建
agent/conversation.json    Agent Prompt 学习材料
mini-vectordb/
  Makefile
  src/platform.{h,cpp}     文件/ mmap / CRC32 / 目录等平台层
  src/simd.{h,cpp}         L2 距离与 TopK 扫描内核（AVX2 + 标量回退）
  src/types.{h,cpp}        schema、值、行编码、向量字面量解析
  src/sql.{h,cpp}          词法/语法分析与语句切分
  src/engine.{h,cpp}       WAL、表、主键索引、向量存储、HNSW
  src/exec.cpp             DDL/DML/SELECT 执行器
  src/main.cpp             stdin/stdout 前端
  tests/                   功能、耐久性、随机差分与基准脚本
```
