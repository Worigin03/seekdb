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
#include "sql/ai_column/ai_column_definition.h"
#include "share/schema/ob_column_schema.h"
#include "lib/hash/ob_hashmap.h"
using namespace oceanbase::common;
using namespace oceanbase::share::schema;
using namespace oceanbase::sql::ai_column;
TEST(AIColumnSchema, definition_copy_and_serialization_own_strings) {
  Definition definition;definition.expression="AI_COMPLETE('model',content)";
  definition.dependencies={"content"};definition.state_column="__ai_state_test";
  definition.request_column="__ai_request_test";definition.result_spec_column="__ai_spec_test";
  definition.result_hint="Return exactly one valid JSON value";definition.max_retries=3;
  std::string json=definition.encode();ObColumnSchemaV2 column;column.set_column_id(500);
  column.set_table_id(500000);column.set_data_type(ObJsonType);
  ASSERT_EQ(OB_SUCCESS,column.set_ai_column_info(ObString(json.size(),json.data())));
  ObColumnSchemaV2 copied;ASSERT_EQ(OB_SUCCESS,copied.assign(column));
  json.assign(json.size(),'x');ASSERT_TRUE(copied.is_ai_column());
  EXPECT_EQ(3,Definition::decode(copied.get_ai_column_info_str()).max_retries);
  std::vector<char> buffer(column.get_serialize_size());int64_t pos=0;
  ASSERT_EQ(OB_SUCCESS,column.serialize(buffer.data(),buffer.size(),pos));EXPECT_EQ(buffer.size(),pos);
  ObColumnSchemaV2 restored;pos=0;ASSERT_EQ(OB_SUCCESS,restored.deserialize(buffer.data(),buffer.size(),pos));
  buffer.assign(buffer.size(),'x');
  EXPECT_EQ(definition.expression,Definition::decode(restored.get_ai_column_info_str()).expression);
  EXPECT_GE(restored.get_convert_size(),sizeof(restored)+restored.get_ai_column_info_str().length());
}
TEST(AIColumnSchema, hidden_columns_and_id_remapping) {
  Definition definition;definition.expression="AI_COMPLETE('model',content)";
  definition.state_column="__ai_state_test";definition.request_column="__ai_request_test";
  definition.result_spec_column="__ai_spec_test";const auto json=definition.encode();
  ObColumnSchemaV2 column;column.set_column_id(500);column.set_table_id(500000);
  ASSERT_EQ(OB_SUCCESS,column.set_ai_column_info(ObString(json.size(),json.data())));
  std::vector<std::unique_ptr<ObColumnSchemaV2>> hidden;
  ASSERT_EQ(OB_SUCCESS,hidden_columns(column,hidden));ASSERT_EQ(3,hidden.size());
  for(const auto &state:hidden)EXPECT_TRUE(state->is_hidden());
  EXPECT_EQ(ObTinyIntType,hidden[0]->get_data_type());EXPECT_EQ(64,hidden[1]->get_data_length());
  hash::ObHashMap<uint64_t,uint64_t> ids;ASSERT_EQ(OB_SUCCESS,ids.create(8,"AIColumnTest"));
  ASSERT_EQ(OB_SUCCESS,ids.set_refactored(500,800));ASSERT_EQ(OB_SUCCESS,column.convert_column_id(ids));
  EXPECT_EQ(800,column.get_column_id());
  EXPECT_EQ(definition.state_column,Definition::decode(column.get_ai_column_info_str()).state_column);
  ids.destroy();
}
