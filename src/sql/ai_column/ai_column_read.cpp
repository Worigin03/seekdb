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
#include "sql/resolver/dml/ob_select_stmt.h"
#include "sql/resolver/dml/ob_update_stmt.h"
#include "sql/session/ob_inner_sql_connection.h"
#include "query/session/ob_inner_sql_connection_access.h"
#include "share/schema/ob_schema_getter_guard.h"
#include "common/mysqlclient/ob_mysql_proxy.h"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include <set>
#include <functional>

namespace oceanbase { namespace sql { namespace ai_column {
using namespace common;
using namespace share::schema;
namespace {
ObString read_string(const std::string &s){return ObString(s.size(),s.data());}
std::string read_text(const ObString &s){return std::string(s.ptr(),s.length());}
}
int capture_read_columns(ObDMLStmt &root,ObSchemaGetterGuard &schema,ObIAllocator &allocator)
{
  int ret=OB_SUCCESS;
  try {
    std::set<std::pair<uint64_t,uint64_t>> references;std::set<const ObDMLStmt *> visited;
    std::function<void(const ObDMLStmt &,const ObRawExpr *,int)> expression;
    std::function<void(const ObDMLStmt &,int)> statement;
    expression=[&](const ObDMLStmt &stmt,const ObRawExpr *expr,int depth) {
      if(OB_FAIL(ret)||!expr)return;if(depth>256){ret=OB_SIZE_OVERFLOW;return;}
      if(expr->is_column_ref_expr()) {
        const auto &column=*static_cast<const ObColumnRefRawExpr *>(expr);
        const auto *item=stmt.get_column_item_by_id(column.get_table_id(),column.get_column_id());
        const auto *table=stmt.get_table_item_by_id(column.get_table_id());
        uint64_t tid=OB_INVALID_ID,cid=column.get_column_id();
        if(item&&item->base_tid_!=OB_INVALID_ID){tid=item->base_tid_;cid=item->base_cid_;}
        else if(table&&table->is_basic_table())tid=table->ref_id_;
        if(tid!=OB_INVALID_ID) {
          const ObTableSchema *definition=nullptr;
          if(OB_FAIL(schema.get_table_schema(tid,definition)))return;
          if(definition){const auto *source=definition->get_column_schema(cid);
            if(source&&source->is_ai_column())references.insert({tid,cid});}
        }
      }
      for(int64_t i=0;OB_SUCC(ret)&&i<expr->get_param_count();++i)expression(stmt,expr->get_param_expr(i),depth+1);
    };
    statement=[&](const ObDMLStmt &stmt,int depth) {
      if(OB_FAIL(ret)||!visited.insert(&stmt).second)return;if(depth>128){ret=OB_SIZE_OVERFLOW;return;}
      for(int64_t i=0;i<stmt.get_condition_exprs().count();++i)expression(stmt,stmt.get_condition_exprs().at(i),0);
      for(int64_t i=0;i<stmt.get_order_items().count();++i)expression(stmt,stmt.get_order_items().at(i).expr_,0);
      for(int64_t i=0;i<stmt.get_joined_tables().count();++i) {
        const auto *joined=stmt.get_joined_tables().at(i);
        if(joined)for(int64_t j=0;j<joined->join_conditions_.count();++j)expression(stmt,joined->join_conditions_.at(j),0);
      }
      if(stmt.is_select_stmt()) {
        const auto &select=static_cast<const ObSelectStmt &>(stmt);
        for(int64_t i=0;i<select.get_select_item_size();++i)expression(stmt,select.get_select_item(i).expr_,0);
        for(int64_t i=0;i<select.get_group_exprs().count();++i)expression(stmt,select.get_group_exprs().at(i),0);
        for(int64_t i=0;i<select.get_having_exprs().count();++i)expression(stmt,select.get_having_exprs().at(i),0);
      } else if(stmt.is_update_stmt()) {
        const auto &update=static_cast<const ObUpdateStmt &>(stmt);
        for(int64_t i=0;i<update.get_update_table_info().count();++i) {
          const auto *table=update.get_update_table_info().at(i);
          if(table)for(int64_t j=0;j<table->assignments_.count();++j)expression(stmt,table->assignments_.at(j).expr_,0);
        }
      }
      ObSEArray<ObSelectStmt *,8> children;
      if(OB_FAIL(stmt.get_child_stmts(children)))return;
      for(auto *child:children)if(child)statement(*child,depth+1);
    };
    statement(root,0);
    if(OB_SUCC(ret)&&!references.empty()) {
      rapidjson::Document data;data.SetArray();auto &a=data.GetAllocator();
      for(const auto &reference:references){rapidjson::Value row(rapidjson::kArrayType);row.PushBack(reference.first,a);row.PushBack(reference.second,a);data.PushBack(row,a);}
      rapidjson::StringBuffer buffer;rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);data.Accept(writer);
      ObString encoded(buffer.GetSize(),buffer.GetString());
      ret=ob_write_string(allocator,encoded,root.get_query_ctx()->ai_column_read_spec_);
    }
  } catch(const std::bad_alloc &){ret=OB_ALLOCATE_MEMORY_FAILED;}
  return ret;
}
int check_strict_read(ObExecContext &ctx,const ObPhysicalPlan &plan)
{
  int ret=OB_SUCCESS;auto *session=ctx.get_my_session();
  if(!session||session->is_ai_column_internal_write()||plan.get_ai_column_read_spec().empty())return OB_SUCCESS;
  bool strict=false;if(OB_FAIL(session->get_sys_variable(share::SYS_VAR_AI_COLUMN_STRICT_READ,strict)))return ret;
  if(!strict)return OB_SUCCESS;
  auto *sql_ctx=ctx.get_sql_ctx();if(!sql_ctx||!sql_ctx->schema_guard_)return OB_ERR_UNEXPECTED;
  try {
    auto encoded=plan.get_ai_column_read_spec();rapidjson::Document references;references.Parse(encoded.ptr(),encoded.length());
    if(references.HasParseError()||!references.IsArray())return OB_INVALID_DATA;
    InternalWriteGuard trusted(*session);sqlclient::ObISQLConnectionGuard connection;
    if(OB_FAIL(query::ObInnerSQLConnectionAccess::create_spi_connection_with_external_session(session,connection)))return ret;
    auto *conn=as_inner_sql_connection(connection.get_ptr());if(!conn)return OB_ERR_UNEXPECTED;
    NestedSession nested(*conn);
    if(OB_FAIL(nested.open()))return ret;
    for(const auto &reference:references.GetArray()) {
      if(OB_FAIL(ret))break;
      if(!reference.IsArray()||reference.Size()!=2||!reference[0].IsUint64()||!reference[1].IsUint64()){ret=OB_INVALID_DATA;break;}
      uint64_t tid=reference[0].GetUint64(),cid=reference[1].GetUint64();
      const ObTableSchema *table=nullptr;const ObDatabaseSchema *database=nullptr;
      if(OB_FAIL(sql_ctx->schema_guard_->get_table_schema(tid,table)))break;
      const auto *column=table?table->get_column_schema(cid):nullptr;
      if(!column||!column->is_ai_column()){ret=OB_SCHEMA_EAGAIN;break;}
      Definition d=Definition::decode(column->get_ai_column_info_str());
      if(OB_FAIL(sql_ctx->schema_guard_->get_database_schema(table->get_database_id(),database)))break;
      if(!database){ret=OB_ERR_UNEXPECTED;break;}
      const auto target=quote_identifier(read_text(database->get_database_name_str()))+"."+quote_identifier(read_text(table->get_table_name_str()));
      const auto state=quote_identifier(d.state_column);
      const auto query="SELECT 1 AS incomplete FROM "+target+" WHERE "+state+" IS NULL OR "+state+"<>1 LIMIT 1";
      ObMySQLProxy::MySQLResult result;
      if(OB_FAIL(conn->execute_read(read_string(query),result)))break;
      if(!result.get_result()){ret=OB_ERR_UNEXPECTED;break;}
      const int next=result.get_result()->next();
      if(next==OB_SUCCESS){ret=OB_AI_COLUMN_NOT_READY;LOG_USER_ERROR(OB_AI_COLUMN_NOT_READY,tid,cid);}
      else if(next!=OB_ITER_END)ret=next;
    }
    const int restore=nested.close();
    if(OB_SUCC(ret))ret=restore;
  } catch(const std::bad_alloc &){ret=OB_ALLOCATE_MEMORY_FAILED;}
    catch(const std::exception &){ret=OB_INVALID_DATA;}
  return ret;
}
}}}
