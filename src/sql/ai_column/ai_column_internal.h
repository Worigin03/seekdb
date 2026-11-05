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
#include "sql/session/ob_inner_sql_connection.h"
namespace oceanbase { namespace sql { namespace ai_column {
class NestedSession {
public:
  explicit NestedSession(ObIInnerSQLConnection &connection):connection_(connection){}
  ~NestedSession(){close();}
  int open(){int ret=connection_.begin_nested_session(session_,connection_state_,false);active_=ret==common::OB_SUCCESS;return ret;}
  int close(){if(!active_)return common::OB_SUCCESS;active_=false;return connection_.end_nested_session(session_,connection_state_);}
private:
  ObIInnerSQLConnection &connection_;
  ObSQLSessionInfo::StmtSavedValue session_;
  ObIInnerSQLConnection::SavedValue connection_state_;
  bool active_=false;
};
}}}
