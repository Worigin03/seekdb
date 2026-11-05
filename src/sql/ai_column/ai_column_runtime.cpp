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
#define USING_LOG_PREFIX SQL_ENG
#include "ai_column_runtime.h"
#include "ai_column_internal.h"
#include "sql/engine/ob_exec_context.h"
#include "sql/engine/ob_physical_plan.h"
#include "sql/ob_sql_context.h"
#include "sql/session/ob_inner_sql_connection.h"
#include "query/session/ob_inner_sql_connection_access.h"
#include "share/schema/ob_schema_getter_guard.h"
#include "share/schema/ob_table_schema.h"
#include "common/mysqlclient/ob_mysql_proxy.h"
#include <set>
#include <algorithm>

namespace oceanbase { namespace sql { namespace ai_column {
using namespace common;
using namespace share::schema;
namespace {
ObString runtime_string(const std::string &s) {return ObString(s.size(),s.data());}
std::string runtime_name(const ObString &s) {return std::string(s.ptr(),s.length());}
std::string runtime_join(const std::vector<std::string> &items,const std::string &separator)
{std::string s;for(const auto &item:items){if(!s.empty())s+=separator;s+=item;}return s;}
std::string runtime_dependency_hash(const std::vector<std::string> &names,const std::string &prefix)
{
  std::vector<std::string> columns;for(const auto &column:names)columns.push_back(prefix+quote_identifier(column));
  return "SHA2(CAST(JSON_ARRAY("+runtime_join(columns,",")+") AS CHAR),256)";
}
}
InternalWriteGuard::InternalWriteGuard(ObSQLSessionInfo &session)
  :session_(session),previous_(session.is_ai_column_internal_write())
{session_.set_ai_column_internal_write(true);}
InternalWriteGuard::~InternalWriteGuard() {session_.set_ai_column_internal_write(previous_);}

int make_queue_plan(ObSchemaGetterGuard &schema,const ObTableSchema &table,
                    const std::string &request,QueuePlan &plan)
{
  int ret=OB_SUCCESS;
  try {
    std::vector<const ObColumnSchemaV2 *> columns;
    for(int64_t i=0;i<table.get_column_count();++i) {
      const auto *column=table.get_column_schema_by_idx(i);
      if(column&&column->is_ai_column())columns.push_back(column);
    }
    if(columns.empty())return OB_SUCCESS;
    const ObDatabaseSchema *database=nullptr;
    if(OB_FAIL(schema.get_database_schema(table.get_database_id(),database)))return ret;
    if(!database)return OB_ERR_UNEXPECTED;
    const std::string target=quote_identifier(runtime_name(database->get_database_name_str()))+"."+quote_identifier(runtime_name(table.get_table_name_str()));
    std::vector<std::string> key_names,predicate_parts;
    for(int64_t i=0;i<table.get_rowkey_column_num();++i) {
      uint64_t key_id = OB_INVALID_ID;
      if (OB_FAIL(table.get_rowkey_info().get_column_id(i, key_id))) return ret;
      const auto *column=table.get_column_schema(key_id);
      if(!column)return OB_ERR_UNEXPECTED;
      const std::string key=runtime_name(column->get_column_name_str());key_names.push_back(key);
      const std::string identifier=quote_identifier(key);
      if(ob_is_integer_type(column->get_data_type())||ob_is_numeric_type(column->get_data_type()))
        predicate_parts.push_back("CONCAT("+sql_string(identifier+" = ")+",CAST("+identifier+" AS CHAR))");
      else predicate_parts.push_back("CONCAT("+sql_string(identifier+" = CONVERT(X'")+",HEX("+identifier+"),"+sql_string("' USING utf8mb4)")+")");
    }
    if(key_names.empty())return OB_NOT_SUPPORTED;
    const std::string rowkey_hash=runtime_dependency_hash(key_names,"r.");
    std::string predicate="CONCAT_WS("+sql_string(" AND ")+","+runtime_join(predicate_parts,",")+")";
    std::vector<std::string> changes,assignments;
    std::vector<Definition> definitions;
    for(const auto *column:columns) {
      Definition d=Definition::decode(column->get_ai_column_info_str());
      auto dependencies=key_names;
      for(const auto &dep:d.dependencies)if(std::find(dependencies.begin(),dependencies.end(),dep)==dependencies.end())dependencies.push_back(dep);
      const std::string expected="SHA2(CONCAT("+sql_string(std::to_string(column->get_schema_version())+":")+","+runtime_dependency_hash(dependencies,"")+"),256)";
      const std::string spec=quote_identifier(d.result_spec_column);
      const std::string changed="("+spec+" IS NULL OR "+spec+" <> "+expected+")";changes.push_back(changed);
      const auto result=quote_identifier(runtime_name(column->get_column_name_str()));
      const auto state=quote_identifier(d.state_column),token=quote_identifier(d.request_column);
      assignments.push_back(result+"=IF("+changed+",NULL,"+result+")");
      assignments.push_back(state+"=IF("+changed+",0,"+state+")");
      const auto request_expr="SHA2(CONCAT("+sql_string(request)+","+runtime_dependency_hash(key_names,"")+"),256)";
      assignments.push_back(token+"=IF("+changed+","+request_expr+","+token+")");
      assignments.push_back(spec+"=IF("+changed+","+expected+","+spec+")");definitions.push_back(std::move(d));
    }
    // A result-only update must preserve business ON UPDATE timestamps.
    for(int64_t i=0;i<table.get_column_count();++i) {
      const auto *column=table.get_column_schema_by_idx(i);
      if(column&&column->is_on_update_current_timestamp()) {
        auto id=quote_identifier(runtime_name(column->get_column_name_str()));assignments.push_back(id+"="+id);
      }
    }
    plan.statements.push_back("UPDATE "+target+" SET "+runtime_join(assignments,",")+" WHERE "+runtime_join(changes," OR "));
    for(size_t i=0;i<columns.size();++i) {
      const auto &d=definitions[i];const auto *column=columns[i];
      const std::string identity="task.table_id="+std::to_string(table.get_table_id())+" AND task.column_id="+std::to_string(column->get_column_id());
      plan.statements.push_back("DELETE task FROM oceanbase.__all_ai_column_task task WHERE "+identity+
          " AND NOT EXISTS(SELECT 1 FROM "+target+" r WHERE task.rowkey_hash="+runtime_dependency_hash(key_names,"r.")+
          " AND task.request_id=r."+quote_identifier(d.request_column)+")");
      const std::string token="r."+quote_identifier(d.request_column);
      const std::string config=d.config_sql.empty()?"NULL":d.config_sql;
      plan.statements.push_back("INSERT INTO oceanbase.__all_ai_column_task "
          "(table_id,column_id,rowkey_hash,request_id,rowkey_predicate,definition_version,state,retry_count,eligible_ts,input,config) SELECT "+
          std::to_string(table.get_table_id())+","+std::to_string(column->get_column_id())+","+rowkey_hash+","+token+","+predicate+","+
          std::to_string(column->get_schema_version())+",0,0,NOW(6),"+d.input_sql+","+config+" FROM "+target+" r"+
          " WHERE "+token+"=SHA2(CONCAT("+sql_string(request)+","+rowkey_hash+"),256) AND "+quote_identifier(d.state_column)+"=0 AND NOT EXISTS(SELECT 1 FROM "
          "oceanbase.__all_ai_column_task task WHERE "+identity+" AND task.rowkey_hash="+rowkey_hash+" AND task.request_id="+token+")");
    }
  } catch(const std::bad_alloc &) {ret=OB_ALLOCATE_MEMORY_FAILED;}
    catch(const std::exception &) {ret=OB_INVALID_ARGUMENT;}
  return ret;
}
static int after_dml_impl(ObExecContext &ctx,const ObPhysicalPlan &physical_plan)
{
  int ret=OB_SUCCESS;auto type=physical_plan.get_stmt_type();
  if(type!=stmt::T_INSERT&&type!=stmt::T_REPLACE&&type!=stmt::T_UPDATE&&type!=stmt::T_DELETE)return OB_SUCCESS;
  auto *session=ctx.get_my_session();auto *sql_ctx=ctx.get_sql_ctx();
  if(session&&(session->is_ai_column_internal_write()||session->get_ddl_info().is_ddl()))return OB_SUCCESS;
  if(!session||!sql_ctx||!sql_ctx->schema_guard_)return OB_ERR_UNEXPECTED;
  std::vector<const ObTableSchema *> tables;std::set<uint64_t> seen;
  const auto &dependencies = physical_plan.get_dependency_table();
  for (int64_t i = 0; i < dependencies.count(); ++i) {
    const auto &dependency = dependencies.at(i);
    if(OB_FAIL(ret))break;
    if(dependency.get_type()!=DEPENDENCY_TABLE||!seen.insert(dependency.get_object_id()).second)continue;
    const ObTableSchema *table=nullptr;
    if(OB_FAIL(sql_ctx->schema_guard_->get_table_schema(dependency.get_object_id(),table))) {}
    else if(table&&table->is_user_table()) {
      for(int64_t i=0;i<table->get_column_count();++i)if(table->get_column_schema_by_idx(i)->is_ai_column()){tables.push_back(table);break;}
    }
  }
  if(OB_FAIL(ret)||tables.empty())return ret;
  InternalWriteGuard trusted(*session);
  sqlclient::ObISQLConnectionGuard connection;
  if(OB_FAIL(query::ObInnerSQLConnectionAccess::create_spi_connection_with_external_session(session,connection)))return ret;
  auto *conn=as_inner_sql_connection(connection.get_ptr());if(!conn)return OB_ERR_UNEXPECTED;
  NestedSession nested(*conn);
  if(OB_FAIL(nested.open()))return ret;
  std::string request;
  {
    ObMySQLProxy::MySQLResult result;ObString value;
    if(OB_FAIL(conn->execute_read(ObString("SELECT UUID() AS request"),result))) {}
    else if(!result.get_result())ret=OB_ERR_UNEXPECTED;
    else if(OB_FAIL(result.get_result()->next())) {}
    else if(OB_FAIL(result.get_result()->get_varchar("request",value))) {}
    else request.assign(value.ptr(),value.length());
  }
  for(const auto *table:tables) {
    if(OB_FAIL(ret))break;QueuePlan queue;
    if(OB_FAIL(make_queue_plan(*sql_ctx->schema_guard_,*table,request,queue)))break;
    for(const auto &statement:queue.statements) {
      int64_t affected=0;
      if(OB_FAIL(conn->execute_write(runtime_string(statement),affected))) {
        LOG_WARN("AI column transactional queue failed",K(ret),"table_id",table->get_table_id());break;
      }
    }
  }
  const int restore=nested.close();
  if(OB_SUCC(ret))ret=restore;
  return ret;
}
int after_dml(ObExecContext &ctx,const ObPhysicalPlan &plan)
{
  try {return after_dml_impl(ctx,plan);}
  catch(const std::bad_alloc &){return OB_ALLOCATE_MEMORY_FAILED;}
  catch(const std::exception &){return OB_ERR_UNEXPECTED;}
}
}}}
