/*
 * Copyright (c) 2025 OceanBase.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gtest/gtest.h>
#include "sql/parser/ob_parser.h"
#include "lib/allocator/ob_allocator.h"
using namespace oceanbase::common;
using namespace oceanbase::sql;
namespace {
const ParseNode *find(const ParseNode *node,ObItemType type) {
  if(!node)return nullptr;if(node->type_==type)return node;
  for(int i=0;i<node->num_child_;++i)if(const auto *found=find(node->children_[i],type))return found;
  return nullptr;
}
}
TEST(FileSqlParser, arguments_are_one_flat_list) {
  for(const char *sql:{"select * from read_csv('/tmp/input.csv')",
      "select * from read_json('/tmp/input.jsonl', sample_size=-1) as f",
      "describe select * from read_parquet('/tmp/input.parquet')"}) {
    ObArenaAllocator allocator;ObParser parser(allocator,DEFAULT_MYSQL_MODE);ParseResult result;
    ASSERT_EQ(OB_SUCCESS,parser.parse(ObString(sql),result)) << sql << ": " << (result.error_msg_ ? result.error_msg_ : "");
    const ParseNode *file=find(result.result_tree_,T_FILE_TABLE);ASSERT_NE(nullptr,file);
    ASSERT_EQ(3,file->num_child_);const ParseNode *args=file->children_[1];
    ASSERT_NE(nullptr,args);EXPECT_EQ(T_EXPR_LIST,args->type_);ASSERT_GE(args->num_child_,1);
    ASSERT_NE(nullptr,args->children_[0]);EXPECT_EQ(T_VARCHAR,args->children_[0]->type_);
  }
}
TEST(FileSqlParser, copy_preserves_query_and_csv_options) {
  ObArenaAllocator allocator;ObParser parser(allocator,DEFAULT_MYSQL_MODE);ParseResult result;
  ASSERT_EQ(OB_SUCCESS,parser.parse(ObString("copy (select 1 as id) to '/tmp/out.csv' (format csv, header, nullstr 'NULL')"),result));
  const ParseNode *copy=find(result.result_tree_,T_COPY_FILE);ASSERT_NE(nullptr,copy);
  ASSERT_EQ(3,copy->num_child_);ASSERT_NE(nullptr,copy->children_[2]);
  EXPECT_EQ(3,copy->children_[2]->num_child_);
}
