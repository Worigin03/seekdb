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
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include "lib/string/ob_string.h"
#include "share/schema/ob_column_schema.h"

struct _ParseNode;
namespace oceanbase { namespace share { namespace schema { class ObTableSchema; } }
namespace sql {
struct ObResolverParams;
class ObRawExpr;
namespace ai_column {

struct Definition {
  int version = 1;
  int max_retries = 2;
  int64_t dimension = 0;
  bool embedding = false;
  bool input_is_json = false;
  std::string expression, normalized_expr, model, input_sql, config_sql, result_hint;
  std::string state_column, request_column, result_spec_column;
  std::vector<std::string> dependencies;
  std::string encode() const;
  static Definition decode(const common::ObString &json);
};

// Capture syntax first; bind after all ordinary columns have their types.
int capture(share::schema::ObColumnSchemaV2 &column, const _ParseNode &attribute);
int bind(ObResolverParams &params, share::schema::ObTableSchema &table,
         share::schema::ObColumnSchemaV2 &column);
int hidden_columns(const share::schema::ObColumnSchemaV2 &column,
                   std::vector<std::unique_ptr<share::schema::ObColumnSchemaV2>> &columns);
int resolve_input(ObResolverParams &params, const share::schema::ObTableSchema &table,
                  const Definition &definition, ObRawExpr *&input);
std::string quote_identifier(const std::string &name);
std::string sql_string(const std::string &value);

}}} // oceanbase::sql::ai_column
