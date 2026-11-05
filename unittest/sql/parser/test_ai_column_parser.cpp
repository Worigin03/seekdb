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
const ParseNode *find_ai(const ParseNode *node,ObItemType type) {
  if(!node)return nullptr;if(node->type_==type)return node;
  for(int i=0;i<node->num_child_;++i)if(const auto *found=find_ai(node->children_[i],type))return found;
  return nullptr;
}
}
TEST(AIColumnParser, create_add_and_modify_preserve_options) {
  for(const char *sql:{
      "create table docs(id bigint primary key, content text, tags json AI COLUMN (AI_COMPLETE('m',content)) WITH(sync_mode='async',max_retries=2))",
      "alter table docs add column emb vector(4) AI COLUMN (AI_EMBED('m',content)) WITH(sync_mode='async')",
      "alter table docs modify column tags json AI COLUMN (AI_COMPLETE('m',concat(content,'!'))) WITH(max_retries=0)"}) {
    ObArenaAllocator allocator;ObParser parser(allocator,DEFAULT_MYSQL_MODE);ParseResult result;
    ASSERT_EQ(OB_SUCCESS,parser.parse(ObString(sql),result)) << sql << ": " << (result.error_msg_?result.error_msg_:"");
    const auto *ai=find_ai(result.result_tree_,T_AI_COLUMN);ASSERT_NE(nullptr,ai);ASSERT_EQ(2,ai->num_child_);
    ASSERT_NE(nullptr,ai->children_[0]);ASSERT_GT(ai->children_[0]->str_len_,0);
    ASSERT_NE(nullptr,ai->children_[1]);ASSERT_GE(ai->children_[1]->num_child_,1);
    EXPECT_EQ(T_OP_EQ,ai->children_[1]->children_[0]->type_);
  }
}
TEST(AIColumnParser, retry_and_show_are_distinct_statements) {
  for(const auto &entry:{std::make_pair("alter table docs retry ai column tags where status='FAILED' and id=1",T_AI_COLUMN_RETRY),
      std::make_pair("alter table docs retry ai column tags",T_AI_COLUMN_RETRY),
      std::make_pair("show ai columns from docs",T_SHOW_AI_COLUMNS)}) {
    ObArenaAllocator allocator;ObParser parser(allocator,DEFAULT_MYSQL_MODE);ParseResult result;
    ASSERT_EQ(OB_SUCCESS,parser.parse(ObString(entry.first),result)) << entry.first << ": " << (result.error_msg_?result.error_msg_:"");
    ASSERT_NE(nullptr,find_ai(result.result_tree_,entry.second));
  }
}
TEST(AIColumnParser, describe_flag_preserves_parameter_counter) {
  ObArenaAllocator allocator;ObParser parser(allocator,DEFAULT_MYSQL_MODE);ParseResult result;
  ASSERT_EQ(OB_SUCCESS,parser.parse(ObString("describe select * from read_csv('/tmp/a.csv')"),result));
  const auto *explain=find_ai(result.result_tree_,T_EXPLAIN);ASSERT_NE(nullptr,explain);
  EXPECT_TRUE(explain->is_file_describe_);EXPECT_EQ(0,explain->value_);
}
