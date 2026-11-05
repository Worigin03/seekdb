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
#include <string>
#include <vector>
#include "ai_column_definition.h"
namespace oceanbase { namespace share { namespace schema { class ObSchemaGetterGuard; } } namespace sql {
class ObExecContext;
class ObPhysicalPlan;
class ObSQLSessionInfo;
class ObDMLStmt;
namespace ai_column {
class InternalWriteGuard {
public:
  explicit InternalWriteGuard(ObSQLSessionInfo &session);
  ~InternalWriteGuard();
private:
  ObSQLSessionInfo &session_;
  bool previous_;
};
struct QueuePlan {
  std::vector<std::string> statements;
};
int make_queue_plan(share::schema::ObSchemaGetterGuard &schema,
                    const share::schema::ObTableSchema &table,
                    const std::string &request, QueuePlan &plan);
int after_dml(ObExecContext &context,const ObPhysicalPlan &plan);
int capture_read_columns(ObDMLStmt &statement,share::schema::ObSchemaGetterGuard &schema,common::ObIAllocator &allocator);
int check_strict_read(ObExecContext &context,const ObPhysicalPlan &plan);
}}}
