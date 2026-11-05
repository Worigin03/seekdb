# seekdb 竞赛功能实现与验证

## 代码入口

| 功能 | 主要入口 |
| --- | --- |
| AI 函数、Provider 和批量 Embedding | `src/sql/engine/expr/ob_expr_ai/` |
| CSV、JSONL、文件发现与 CSV 导出 | `src/sql/engine/basic/file_scan_utils.{h,cpp}` |
| Parquet Footer、Page、字典及 Snappy | `src/sql/engine/basic/parquet_reader.{h,cpp}` |
| 文件 SQL 的原生扫描算子 | `src/sql/engine/basic/ob_function_table_op.cpp` |
| COPY 的原生执行入口 | `src/sql/engine/basic/ob_select_into_op.cpp` |
| AI 列规则绑定、事务入队及严格读取 | `src/sql/ai_column/` |
| AI 列后台领取、重试和分组发布 | `src/observer/ai_service/ai_column_service.cpp` |
| SQL 语法 | `src/sql/parser/sql_parser_mysql_mode.y` |

## 文件 SQL

`enable_file_sql` 默认关闭。开启后，解析和执行阶段仍校验 `secure_file_priv`。启动实例时可通过 `--variable secure_file_priv=/允许访问的目录` 配置路径范围。

CSV、JSONL 先流式扫描全部记录推断 Schema，执行时再次流式扫描；`sample_size` 只接受 `-1`。投影、过滤、聚合、排序和 JOIN 由 seekdb 优化器与执行器完成。Parquet 根据 Footer 创建列信息，仅为投影列解码数据页，支持未压缩和 Snappy、PLAIN 和字典编码、V1/V2 数据页。

扫描记录文件的设备号、inode、长度、纳秒修改时间及状态变化时间，并在打开、取数、结束和重扫时检查。路径解析检查允许目录边界，随后使用逐级 `openat` 和 `O_NOFOLLOW` 打开文件。CSV 导出使用同目录独占临时文件；查询完成后原子发布，目标已经存在时失败，失败路径清理临时文件。

资源限制包括：最多 4096 列、列名最多 256 字节、字段最多 16 MiB、CSV/JSONL 记录最多 64 MiB、glob 最多 100000 个结果。Parquet 对 Footer、页、值数量、字典和保留内存分别设限，拒绝不支持的编码、压缩方式和嵌套结构。全 NULL 列保留 NULL 类型。

## 异步 AI 列

规则随 `ObColumnSchemaV2` 持久化，包括原表达式、规范化表达式、模型、输入与配置表达式、依赖列、类型提示词和重试配置。每个结果列配有隐藏状态、请求标识和依赖指纹列。`oceanbase.__all_ai_column` 提供规则及隐藏列 ID 元数据；持久化任务位于 `oceanbase.__all_ai_column_task`。

业务 DML 关闭执行计划后、结束语句和提交事务前，在同一事务内使旧结果失效并创建任务。回滚同时撤销数据、结果失效和任务写入。当前入队实现按语句扫描涉及的 AI 表，以主键、依赖值及定义版本的指纹判断变化；批量 DML 按语句处理，但单行写入大表时仍有整表检查成本。

后台线程只读取已提交任务，领取同一行、同一请求的一组列，模型调用发生在发布事务之外。模型结果先按目标类型检查并持久化，再以一个事务发布该组的结果与终态。写回检查任务租约、请求标识和列定义版本，并先锁业务行再锁任务行。删除及主键变化清理旧任务，迟到响应不能恢复已删除行或覆盖新请求。成功、失败列可以在同一组中共同发布。

隐藏状态值为 `0=PENDING`、`1=READY`、`2=FAILED`。NULL 输入可以得到 READY 状态和 SQL NULL 结果，完成与否由状态判断。`ai_column_strict_read` 按语句实际引用的 AI 列检查整列就绪情况。`ALTER TABLE ... RETRY AI COLUMN ... WHERE ...` 支持状态伪列；反引号中的 `status` 保持业务列含义。

后台写入使用原生 DML，从而复用普通索引及向量索引维护和事务回滚机制。内部维护计划禁止进入共享计划缓存，业务语句不能直接赋值 AI 结果列。

## 本地验证入口

在 Linux 环境按仓库依赖安装流程准备构建工具后：

```bash
bash build.sh release -DOB_MAX_UNITY_BATCH_SIZE=5 -DENABLE_THIN_LTO=OFF
make -C build_release -j2 seekdb ai_function_tests file_scan_tests file_sql_parser_tests ai_column_schema_tests
ctest --test-dir build_release/unittest --output-on-failure
```

补充集成回归连接已经启动的隔离测试实例。文件 SQL 的 `--root` 必须存在，且位于该实例的 `secure_file_priv` 范围内：

```bash
python3 unittest/sql/engine/basic/test_file_sql_integration.py \
  --client deps/3rd/u01/obclient/bin/obclient --port 2888 --root /允许访问的目录 -v

python3 unittest/sql/engine/expr/test_ai_column_integration.py \
  --client deps/3rd/u01/obclient/bin/obclient --port 2888 \
  --mock tools/deploy/mysql_test/test_suite/ai_column/include/ai_column_mock.py -v
```

AI 列补充回归使用本地端口 19125 的公开 Mock 实现，不需要真实模型凭据；建库、模型及 Endpoint 名称使用随机后缀。覆盖 NULL 完成状态、直接赋值保护、长文本输入与结果、配置表达式限制、部分请求失效和规则修改后的重算。

公开 mysqltest 仍位于 `tools/deploy/mysql_test/test_suite/{ai_function,file_sql,ai_column}/`，测试脚本与预期结果保持原样。通过本地测试不代表隐藏用例或代码评测得分，最终结果以评测系统回传为准。
