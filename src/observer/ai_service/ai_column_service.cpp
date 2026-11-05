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
#define USING_LOG_PREFIX SERVER
#include "ai_column_service.h"
#include "observer/ob_inner_sql_connection.h"
#include "observer/ob_inner_sql_result.h"
#include "sql/ob_result_set.h"
#include "sql/engine/expr/ob_expr_lob_utils.h"
#include "sql/ai_column/ai_column_runtime.h"
#include "sql/engine/expr/ob_expr_ai/ob_ai_func_utils.h"
#include "query/ai/ob_ai_endpoint_resolver.h"
#include "query/session/ob_inner_sql_connection_access.h"
#include "share/ob_server_struct.h"
#include "share/rc/ob_server_runtime.h"
#include "share/schema/ob_multi_version_schema_service.h"
#include "share/schema/ob_schema_getter_guard.h"
#include "common/mysqlclient/ob_mysql_proxy.h"
#include "lib/charset/ob_charset.h"
#include "rapidjson/document.h"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>
#include <unistd.h>

namespace oceanbase { namespace observer {
using namespace common;
using namespace share::schema;
using namespace sql::ai_column;
namespace {
constexpr size_t MAX_TASK_BYTES=16*1024*1024;
const char *TASKS="oceanbase.__all_ai_column_task";
ObString worker_string(const std::string &s){return ObString(s.size(),s.data());}
std::string worker_text(const ObString &s){return std::string(s.ptr(),s.length());}
int write_task_sql(sqlclient::ObISQLConnection &connection,const std::string &sql,int64_t *count=nullptr)
{
  auto &inner=static_cast<ObInnerSQLConnection &>(connection);
  const bool standalone=!inner.is_in_trans();
  int64_t affected=0;
  int ret=standalone?connection.start_transaction():OB_SUCCESS;
  if(OB_SUCC(ret)) {
    ret=connection.execute_write(worker_string(sql),affected);
    if(standalone) {
      if(OB_SUCC(ret))ret=connection.commit();
      else connection.rollback();
    }
  }
  if(count)*count=OB_SUCC(ret)?affected:0;
  return ret;
}
int read_task_field(sqlclient::ObMySQLResult &row,const char *column,std::string &value,bool *is_null=nullptr)
{
  ObObj object;ObString s;int ret=row.get_obj(column,object);
  if(ret==OB_ERR_NULL_VALUE){if(is_null)*is_null=true;value.clear();return OB_SUCCESS;}
  if(OB_SUCC(ret)&&object.is_null()){if(is_null)*is_null=true;value.clear();return OB_SUCCESS;}
  ObArenaAllocator allocator("AIColumnRead");
  if(OB_SUCC(ret)&&object.is_lob_storage()) {
    ObString locator_bytes=object.get_string();
    ObLobLocatorV2 locator(locator_bytes,object.has_lob_header());int64_t length=0;
    if(!locator.is_valid(false))ret=OB_INVALID_DATA;
    else if(OB_FAIL(locator.get_lob_data_byte_len(length))) {}
    else if(length<0||length>MAX_TASK_BYTES)ret=OB_SIZE_OVERFLOW;
    else {
      // All service connections are local ObInnerSQLConnections. Read out-of-row
      // task payloads under the result's snapshot instead of treating a locator
      // as the model input (or silently returning an empty string).
      auto &context=static_cast<ObInnerSQLResult &>(row).result_set().get_exec_context();
      ret=sql::ObTextStringHelper::read_real_string_data(context,&allocator,object,s);
    }
  } else if(OB_SUCC(ret))ret=object.get_varchar(s);
  if(OB_SUCC(ret)) {if(s.length()>MAX_TASK_BYTES)return OB_SIZE_OVERFLOW;value=worker_text(s);if(is_null)*is_null=false;}
  return ret;
}
std::string worker_join(const std::vector<std::string> &items,const char *separator)
{std::string result;for(const auto &item:items){if(!result.empty())result+=separator;result+=item;}return result;}
struct TaskGroup {
  uint64_t table_id=0;std::string rowkey_hash,request,owner;
  std::string predicate() const {
    return "table_id="+std::to_string(table_id)+" AND rowkey_hash="+sql_string(rowkey_hash)+" AND request_id="+sql_string(request);
  }
  std::string owned() const {return predicate()+" AND lease_owner="+sql_string(owner);}
};
struct ColumnTask {
  uint64_t column_id=0;int64_t definition_version=0,state=0,retries=0;
  std::string predicate,input,config,result;bool input_null=false,result_null=false;
};
int make_request_uuid(sqlclient::ObISQLConnection &connection,std::string &value)
{
  ObMySQLProxy::MySQLResult result;int ret=connection.execute_read(ObString("SELECT UUID() AS id"),result);
  if(OB_SUCC(ret)&&!result.get_result())ret=OB_ERR_UNEXPECTED;
  if(OB_SUCC(ret))ret=result.get_result()->next();
  if(OB_SUCC(ret))ret=read_task_field(*result.get_result(),"id",value);
  return ret;
}
int claim_task_group(sqlclient::ObISQLConnection &connection,TaskGroup &group,bool &found)
{
  int ret=OB_SUCCESS;found=false;
  {
    ObMySQLProxy::MySQLResult result;
    const std::string query=std::string("SELECT table_id,rowkey_hash,request_id FROM ")+TASKS+" t WHERE "
      "(lease_owner IS NULL OR lease_expire<NOW(6)) AND ((state IN(0,3) AND eligible_ts<=NOW(6)) OR state=1 OR "
      "(state IN(4,5) AND NOT EXISTS(SELECT 1 FROM "+TASKS+" p WHERE p.table_id=t.table_id AND p.rowkey_hash=t.rowkey_hash "
      "AND p.request_id=t.request_id AND p.state NOT IN(4,5)))) ORDER BY eligible_ts,table_id LIMIT 1";
    if(OB_FAIL(connection.execute_read(worker_string(query),result))) {}
    else if(!result.get_result())ret=OB_ERR_UNEXPECTED;
    else if((ret=result.get_result()->next())==OB_ITER_END)ret=OB_SUCCESS;
    else if(OB_SUCC(ret)) {
      int64_t id=0;
      if(OB_FAIL(result.get_result()->get_int("table_id",id))) {}
      else if(OB_FAIL(read_task_field(*result.get_result(),"rowkey_hash",group.rowkey_hash))) {}
      else if(OB_FAIL(read_task_field(*result.get_result(),"request_id",group.request))) {}
      else {group.table_id=id;found=true;}
    }
  }
  if(OB_SUCC(ret)&&found)ret=make_request_uuid(connection,group.owner);
  if(OB_SUCC(ret)&&found) {
    int64_t affected=0;
    ret=write_task_sql(connection,std::string("UPDATE ")+TASKS+" SET lease_owner="+sql_string(group.owner)+
      ",lease_expire=DATE_ADD(NOW(6),INTERVAL 120 SECOND),state=IF(state IN(0,1,3) AND eligible_ts<=NOW(6),1,state) WHERE "+group.predicate()+
      " AND (lease_owner IS NULL OR lease_expire<NOW(6))",&affected);
    found=OB_SUCC(ret)&&affected>0;
  }
  return ret;
}
int load_task_group(sqlclient::ObISQLConnection &connection,const TaskGroup &group,std::vector<ColumnTask> &tasks)
{
  int ret=OB_SUCCESS;ObMySQLProxy::MySQLResult result;
  const auto query=std::string("SELECT column_id,definition_version,state,retry_count,rowkey_predicate,input,config,result FROM ")+TASKS+
    " WHERE "+group.owned()+" ORDER BY column_id";
  if(OB_FAIL(connection.execute_read(worker_string(query),result)))return ret;
  if(!result.get_result())return OB_ERR_UNEXPECTED;
  size_t bytes=0;
  while(OB_SUCC(ret)&&(ret=result.get_result()->next())==OB_SUCCESS) {
    if(tasks.size()>=64)return OB_SIZE_OVERFLOW;
    ColumnTask task;int64_t id=0;auto &row=*result.get_result();
    if(OB_FAIL(row.get_int("column_id",id))) {}
    else if(OB_FAIL(row.get_int("definition_version",task.definition_version))) {}
    else if(OB_FAIL(row.get_int("state",task.state))) {}
    else if(OB_FAIL(row.get_int("retry_count",task.retries))) {}
    else if(OB_FAIL(read_task_field(row,"rowkey_predicate",task.predicate))) {}
    else if(OB_FAIL(read_task_field(row,"input",task.input,&task.input_null))) {}
    else if(OB_FAIL(read_task_field(row,"config",task.config))) {}
    else if(OB_FAIL(read_task_field(row,"result",task.result,&task.result_null))) {}
    else {task.column_id=id;bytes+=task.input.size()+task.config.size()+task.result.size();
      if(bytes>64*1024*1024)ret=OB_SIZE_OVERFLOW;else tasks.push_back(std::move(task));}
  }
  return ret==OB_ITER_END?OB_SUCCESS:ret;
}
bool decimal_result_fits(const std::string &text,int precision,int scale)
{
  if(precision<=0||scale<0||scale>precision)return true;
  size_t begin=!text.empty()&&(text[0]=='+'||text[0]=='-')?1:0;
  const size_t exponent_at=text.find_first_of("eE",begin);
  const size_t end=exponent_at==std::string::npos?text.size():exponent_at;
  size_t dot=text.find('.',begin);if(dot==std::string::npos||dot>end)dot=end;
  int64_t exponent=0;
  if(exponent_at!=std::string::npos) {
    size_t i=exponent_at+1;bool negative=i<text.size()&&text[i]=='-';
    if(i<text.size()&&(text[i]=='+'||text[i]=='-'))++i;
    for(;i<text.size();++i)exponent=std::min<int64_t>(10000,exponent*10+text[i]-'0');
    if(negative)exponent=-exponent;
  }
  std::string digits;
  for(size_t i=begin;i<end;++i)if(text[i]!='.')digits+=text[i];
  size_t zeros=0;while(zeros<digits.size()&&digits[zeros]=='0')++zeros;
  if(zeros==digits.size())return true;
  const int64_t point=static_cast<int64_t>(dot-begin)+exponent-zeros;
  const int64_t capacity=precision-scale;
  if(point!=capacity)return point<capacity;
  // Overflow can also arise when scale rounding carries through all retained
  // digits. Compare decimal digits directly, without losing BIG DECIMAL precision.
  for(int i=0;i<precision;++i)
    if(zeros+i>=digits.size()||digits[zeros+i]!='9')return true;
  return zeros+precision>=digits.size()||digits[zeros+precision]<'5';
}
int validate_result(const ObColumnSchemaV2 &column,const Definition &d,const ObString &value)
{
  if(value.length()>MAX_TASK_BYTES)return OB_SIZE_OVERFLOW;
  if(column.is_json()||d.embedding) {
    rapidjson::Document parsed;parsed.Parse<rapidjson::kParseValidateEncodingFlag>(value.ptr(),value.length());
    if(parsed.HasParseError())return OB_INVALID_DATA;
    if(d.embedding) {
      if(!parsed.IsArray()||parsed.Size()!=d.dimension)return OB_ERR_INVALID_VECTOR_DIM;
      for(const auto &item:parsed.GetArray())if(!item.IsNumber()||!std::isfinite(item.GetDouble())||!std::isfinite(static_cast<float>(item.GetDouble())))return OB_INVALID_DATA;
    }
  } else if(ob_is_integer_type(column.get_data_type())) {
    std::string s=worker_text(value);char *end=nullptr;errno=0;
    if(ob_is_unsigned_type(column.get_data_type())) {
      if(!s.empty()&&s[0]=='-')return OB_INVALID_DATA;
      const auto v=std::strtoull(s.c_str(),&end,10);
      unsigned bits=column.get_data_type()==ObUTinyIntType?8:column.get_data_type()==ObUSmallIntType?16:
        column.get_data_type()==ObUMediumIntType?24:column.get_data_type()==ObUInt32Type?32:64;
      if(errno||end==s.c_str()||*end||(bits<64&&v>=(UINT64_C(1)<<bits)))return OB_INVALID_DATA;
    } else {
      const auto v=std::strtoll(s.c_str(),&end,10);
      unsigned bits=column.get_data_type()==ObTinyIntType?8:column.get_data_type()==ObSmallIntType?16:
        column.get_data_type()==ObMediumIntType?24:column.get_data_type()==ObInt32Type?32:64;
      if(errno||end==s.c_str()||*end||(bits<64&&(v<-(INT64_C(1)<<(bits-1))||v>=(INT64_C(1)<<(bits-1)))))return OB_INVALID_DATA;
    }
  } else if(ob_is_numeric_type(column.get_data_type())) {
    std::string s=worker_text(value);char *end=nullptr;errno=0;double v=std::strtod(s.c_str(),&end);
    if(errno||end==s.c_str()||*end||!std::isfinite(v))return OB_INVALID_DATA;
    for(char c:s)if((c<'0'||c>'9')&&c!='+'&&c!='-'&&c!='.'&&c!='e'&&c!='E')return OB_INVALID_DATA;
    if(ob_is_unsigned_type(column.get_data_type())&&v<0)return OB_INVALID_DATA;
    if((column.get_data_type()==ObFloatType||column.get_data_type()==ObUFloatType)
       &&!std::isfinite(static_cast<float>(v)))return OB_INVALID_DATA;
    if((column.get_data_type()==ObNumberType||column.get_data_type()==ObUNumberType||column.get_data_type()==ObDecimalIntType)
       &&!decimal_result_fits(s,column.get_accuracy().get_precision(),column.get_accuracy().get_scale()))return OB_INVALID_DATA;
  } else if(ob_is_string_type(column.get_data_type())) {
    int64_t well=0;
    if(ObCharset::well_formed_len(column.get_collation_type(),value.ptr(),value.length(),well)!=OB_SUCCESS||well!=value.length())return OB_INVALID_DATA;
    if(ObCharset::strlen_char(column.get_collation_type(),value.ptr(),value.length())>column.get_data_length())return OB_SIZE_OVERFLOW;
  }
  return OB_SUCCESS;
}
int invoke_model(const ObColumnSchemaV2 &column,const Definition &d,const ColumnTask &task,ObArenaAllocator &allocator,std::string &output)
{
  int ret=OB_SUCCESS;ObAIFuncExprInfo *info=nullptr;share::ObAiModelEndpointInfo endpoint;
  auto *resolver=share::server_service<query::ObIAiEndpointResolver>();
  ObString input=worker_string(task.input),result;ObJsonObject *config=nullptr;
  if(task.input_null){output.clear();return OB_SUCCESS;}
  if(d.input_is_json) {
    ObJsonObject *prompt=nullptr;
    if(OB_FAIL(ObAIFuncJsonUtils::get_json_object_form_str(allocator,input,prompt)))return ret;
    if(!ObAIFuncPromptObjectUtils::is_valid_prompt_object(prompt))return OB_INVALID_ARGUMENT;
    if(OB_FAIL(ObAIFuncPromptObjectUtils::replace_all_str_args_in_template(allocator,prompt,input)))return ret;
  }
  if(!task.config.empty()) {
    ObString value=worker_string(task.config);
    if(OB_FAIL(ObAIFuncJsonUtils::get_json_object_form_str(allocator,value,config)))return ret;
    for(const char *reserved:{"model","messages","prompt","input","system"}) {
      ObIJsonBase *node=nullptr;const int lookup=config->get_object_value(ObString(reserved),node);
      if(lookup==OB_SUCCESS)return OB_INVALID_ARGUMENT;
    }
  }
  if(d.embedding) {
    if(OB_FAIL(ObAIFuncJsonUtils::get_json_object(allocator,config)))return ret;
    ObJsonInt *dimension=nullptr;
    if(OB_FAIL(ObAIFuncJsonUtils::get_json_int(allocator,d.dimension,dimension)))return ret;
    if(OB_FAIL(config->add("dimensions",dimension)))return ret;
  }
  ObString model=worker_string(d.model);
  if(!resolver)return OB_NOT_INIT;
  if(OB_FAIL(ObAIFuncUtils::get_ai_func_info(allocator,model,info)))return ret;
  if(OB_FAIL(resolver->resolve_by_model_name(model,allocator,endpoint)))return ret;
  // The queue persists and budgets retries for each task.
  ObAIFuncModel service(allocator,*info,endpoint,false);
  if(d.embedding)ret=service.call_dense_embedding(input,config,result);
  else ret=service.call_completion(input,config,result,worker_string(d.result_hint));
  if(OB_SUCC(ret))ret=validate_result(column,d,result);
  if(OB_SUCC(ret))output=worker_text(result);
  return ret;
}
int publish_group(sqlclient::ObISQLConnection &connection,const TaskGroup &group,const std::vector<ColumnTask> &tasks)
{
  int ret=OB_SUCCESS;ObSchemaGetterGuard schema;const ObTableSchema *table=nullptr;
  if(OB_FAIL(GCTX.schema_service_->get_runtime_schema_guard(schema)))return ret;
  if(OB_FAIL(schema.get_table_schema(group.table_id,table)))return ret;
  if(!table)return write_task_sql(connection,std::string("DELETE FROM ")+TASKS+" WHERE "+group.owned());
  const ObDatabaseSchema *database=nullptr;
  if(OB_FAIL(schema.get_database_schema(table->get_database_id(),database)))return ret;
  if(!database)return OB_ERR_UNEXPECTED;
  const auto target=quote_identifier(worker_text(database->get_database_name_str()))+"."+quote_identifier(worker_text(table->get_table_name_str()));
  std::vector<size_t> valid;std::vector<Definition> definitions(tasks.size());std::vector<std::string> token_exprs;
  for(size_t i=0;i<tasks.size();++i) {
    const auto *column=table->get_column_schema(tasks[i].column_id);
    if(column&&column->is_ai_column()&&column->get_schema_version()==tasks[i].definition_version) {
      definitions[i]=Definition::decode(column->get_ai_column_info_str());valid.push_back(i);
      token_exprs.push_back(quote_identifier(definitions[i].request_column)+" AS token_"+std::to_string(i));
    }
  }
  if(OB_FAIL(connection.start_transaction()))return ret;
  std::vector<std::string> assignments;
  if(!valid.empty()) {
    ObMySQLProxy::MySQLResult result;
    const auto query="SELECT "+worker_join(token_exprs,",")+" FROM "+target+" WHERE "+tasks.front().predicate+" FOR UPDATE";
    if(OB_FAIL(connection.execute_read(worker_string(query),result))) {}
    else if(!result.get_result())ret=OB_ERR_UNEXPECTED;
    else if((ret=result.get_result()->next())==OB_ITER_END)ret=OB_SUCCESS;
    else if(OB_SUCC(ret)) {
      for(auto i:valid) {
        std::string token;
        if(OB_FAIL(read_task_field(*result.get_result(),("token_"+std::to_string(i)).c_str(),token)))break;
        if(token!=group.request)continue;
        const auto *column=table->get_column_schema(tasks[i].column_id);const auto &d=definitions[i];
        const std::string value=tasks[i].state==5||tasks[i].result_null?"NULL":sql_string(tasks[i].result);
        assignments.push_back(quote_identifier(worker_text(column->get_column_name_str()))+"="+value);
        assignments.push_back(quote_identifier(d.state_column)+"="+(tasks[i].state==4?"1":"2"));
      }
    }
  }
  // Lock task rows after the business row, matching the DML enqueue lock order.
  if(OB_SUCC(ret)) {
    ObMySQLProxy::MySQLResult locked;
    ret=connection.execute_read(worker_string(std::string("SELECT column_id FROM ")+TASKS+" WHERE "+group.owned()+" FOR UPDATE"),locked);
    size_t count=0;
    if(OB_SUCC(ret)&&!locked.get_result())ret=OB_ERR_UNEXPECTED;
    while(OB_SUCC(ret)&&(ret=locked.get_result()->next())==OB_SUCCESS)++count;
    if(ret==OB_ITER_END)ret=count==tasks.size()?OB_SUCCESS:OB_STATE_NOT_MATCH;
  }
  if(OB_SUCC(ret)&&!assignments.empty()) {
    for(int64_t i=0;i<table->get_column_count();++i) {
      const auto *column=table->get_column_schema_by_idx(i);
      if(column->is_on_update_current_timestamp()){auto id=quote_identifier(worker_text(column->get_column_name_str()));assignments.push_back(id+"="+id);}
    }
    ret=write_task_sql(connection,"UPDATE "+target+" SET "+worker_join(assignments,",")+" WHERE "+tasks.front().predicate);
  }
  if(OB_SUCC(ret))ret=write_task_sql(connection,std::string("DELETE FROM ")+TASKS+" WHERE "+group.owned());
  if(OB_SUCC(ret))ret=connection.commit();else connection.rollback();
  return ret;
}
int process_group(sqlclient::ObISQLConnection &connection,const TaskGroup &group)
{
  int ret=OB_SUCCESS;std::vector<ColumnTask> tasks;
  if(OB_FAIL(load_task_group(connection,group,tasks)))return ret;
  if(tasks.empty())return OB_SUCCESS;
  ObSchemaGetterGuard schema;const ObTableSchema *table=nullptr;
  if(OB_FAIL(GCTX.schema_service_->get_runtime_schema_guard(schema)))return ret;
  if(OB_FAIL(schema.get_table_schema(group.table_id,table)))return ret;
  if(!table)return write_task_sql(connection,std::string("DELETE FROM ")+TASKS+" WHERE "+group.owned());
  bool terminal=true;
  for(auto &task:tasks) {
    if(task.state==4||task.state==5)continue;
    if(task.state==3){terminal=false;continue;}
    int64_t owned=0;
    if(OB_FAIL(write_task_sql(connection,std::string("UPDATE ")+TASKS+" SET lease_expire=DATE_ADD(NOW(6),INTERVAL 120 SECOND) WHERE "+group.owned(),&owned)))return ret;
    if(!owned)return OB_STATE_NOT_MATCH;
    const auto *column=table->get_column_schema(task.column_id);Definition d;
    int error=OB_SCHEMA_EAGAIN;ObArenaAllocator allocator("AIColumn");
    if(column&&column->is_ai_column()&&column->get_schema_version()==task.definition_version) {
      d=Definition::decode(column->get_ai_column_info_str());error=invoke_model(*column,d,task,allocator,task.result);
    }
    task.result_null=task.input_null;
    if(error==OB_SUCCESS)task.state=4;
    else if(task.retries<d.max_retries&&error!=OB_SCHEMA_EAGAIN){task.state=3;++task.retries;terminal=false;}
    else {
      task.state=5;
      task.result_null=true;
      LOG_WARN_RET(error,"AI column task reached terminal failure",
                   "table_id",group.table_id,"column_id",task.column_id,"retries",task.retries);
    }
    const auto update=std::string("UPDATE ")+TASKS+" SET state="+std::to_string(task.state)+",retry_count="+std::to_string(task.retries)+
      ",eligible_ts="+(task.state==3?"DATE_ADD(NOW(6),INTERVAL 1 SECOND)":"NOW(6)")+",result="+
      (task.result_null?"NULL":sql_string(task.result))+",last_error="+sql_string(std::to_string(error))+" WHERE "+group.owned()+
      " AND column_id="+std::to_string(task.column_id);
    if(OB_FAIL(write_task_sql(connection,update,&owned)))return ret;
    if(!owned)return OB_STATE_NOT_MATCH;
  }
  if(terminal)ret=publish_group(connection,group,tasks);
  if(!terminal||OB_FAIL(ret)) {
    const int released=write_task_sql(connection,std::string("UPDATE ")+TASKS+" SET lease_owner=NULL,lease_expire=NULL WHERE "+group.owned());
    if(OB_SUCC(ret))ret=released;
  }
  return ret;
}
int backfill_ai_columns(sqlclient::ObISQLConnection &connection, int64_t &last_schema_version)
{
  int ret = OB_SUCCESS;
  ObSchemaGetterGuard schema;
  int64_t schema_version = OB_INVALID_VERSION;
  ObSEArray<const ObSimpleTableSchemaV2 *, 64> tables;
  std::string request;
  if (OB_FAIL(GCTX.schema_service_->get_runtime_schema_guard(schema))) {
  } else if (OB_FAIL(schema.get_schema_version(schema_version))) {
  } else if (schema_version == last_schema_version) {
  } else if (OB_FAIL(schema.get_table_schemas_in_runtime(tables))) {
  } else if (OB_FAIL(make_request_uuid(connection, request))) {
  } else {
    int backfill_error = OB_SUCCESS;
    for (const auto *entry : tables) {
      if (entry == nullptr || !entry->is_user_table() || entry->is_in_recyclebin()) {
        continue;
      }
      const ObTableSchema *table = nullptr;
      QueuePlan queue;
      if (OB_FAIL(schema.get_table_schema(entry->get_table_id(), table))) {
      } else if (table == nullptr) {
        continue;
      } else if (OB_FAIL(make_queue_plan(schema, *table, request, queue))) {
      } else if (queue.statements.empty()) {
        continue;
      } else if (OB_FAIL(connection.start_transaction())) {
      } else {
        for (const auto &statement : queue.statements) {
          if (OB_FAIL(write_task_sql(connection, statement))) {
            break;
          }
        }
        if (OB_SUCC(ret)) {
          ret = connection.commit();
        } else {
          connection.rollback();
        }
      }
      if (OB_FAIL(ret)) {
        LOG_WARN("AI column definition backfill failed", K(ret), "table_id", entry->get_table_id());
        backfill_error = ret;
        ret = OB_SUCCESS;
      }
    }
    ret = backfill_error;
    if (OB_SUCC(ret)) {
      last_schema_version = schema_version;
    }
  }
  return ret;
}
}
AiColumnService &AiColumnService::instance(){static AiColumnService service;return service;}
int AiColumnService::start_service()
{
  std::lock_guard<std::mutex> lock(mutex_);if(started_)return OB_SUCCESS;
  recovered_.store(false);
  set_run_wrapper(share::server_runtime());int ret=set_thread_count(4);
  if(OB_SUCC(ret))ret=start();if(OB_SUCC(ret))started_=true;return ret;
}
void AiColumnService::stop_service()
{std::lock_guard<std::mutex> lock(mutex_);if(started_){stop();wait();started_=false;}}
void AiColumnService::run1()
{
  sqlclient::ObISQLConnectionGuard guard;
  // The first argument also sets the inner session's DDL execution flag.
  // Queue maintenance requires ordinary transactional DML plans.
  const int open_ret=ObInnerSQLConnection::create_connection_with_owned_session(false,0,guard);
  if(open_ret!=OB_SUCCESS||!guard.get_ptr()){
    LOG_ERROR_RET(open_ret,"failed to open AI column worker connection");
    return;
  }
  auto &connection=*guard.get_ptr();auto *session=query::ObInnerSQLConnectionAccess::get_session(&connection);
  if(!session)return;
  // Individual leases and inference outcomes must be durable before the next
  // statement. Backfill and group publication open explicit transactions.
  const int autocommit_ret=connection.set_session_variable(ObString("autocommit"),int64_t(1));
  if(autocommit_ret!=OB_SUCCESS){
    LOG_ERROR_RET(autocommit_ret,"failed to enable AI column worker autocommit");
    return;
  }
  InternalWriteGuard trusted(*session);
  int64_t last_backfill = 0;
  int64_t backfilled_schema_version = OB_INVALID_VERSION;
  while(!has_set_stop()) {
    if(!share::server_runtime()->is_write_enabled()){usleep(100000);continue;}
    try {
      if(!recovered_.load()) {
        if(get_thread_idx()==0) {
          // This is the single local process: leases from the previous process
          // can be released immediately, including terminal results awaiting publication.
          const int recovery=write_task_sql(connection,std::string("UPDATE ")+TASKS+
              " SET lease_owner=NULL,lease_expire=NULL,state=IF(state=1,0,state) WHERE lease_owner IS NOT NULL");
          if(recovery==OB_SUCCESS)recovered_.store(true);
          else LOG_WARN_RET(recovery,"AI column task recovery failed");
        }
        if(!recovered_.load()){usleep(100000);continue;}
      }
      if (get_thread_idx() == 0 && ObTimeUtility::current_time() - last_backfill > 5000000) {
        const int backfill_ret = backfill_ai_columns(connection, backfilled_schema_version);
        if (backfill_ret != OB_SUCCESS) {
          LOG_WARN_RET(backfill_ret, "AI column backfill iteration failed");
        }
        last_backfill = ObTimeUtility::current_time();
      }
      TaskGroup group;bool found=false;int ret=claim_task_group(connection,group,found);
      if(OB_SUCC(ret)&&found)ret=process_group(connection,group);
      if(found&&ret!=OB_SUCCESS&&!group.owner.empty()) {
        // A concurrent update may remove only some members of a claimed group.
        // Release the unaffected members even when processing exited early;
        // completed payloads remain durable and do not need another inference.
        const int released=write_task_sql(connection,std::string("UPDATE ")+TASKS+
            " SET lease_owner=NULL,lease_expire=NULL,state=IF(state=1,0,state) WHERE "+group.owned());
        if(released!=OB_SUCCESS)LOG_WARN("AI column lease release failed",K(released));
      }
      if(ret!=OB_SUCCESS&&ret!=OB_STATE_NOT_MATCH)LOG_WARN("AI column worker iteration failed",K(ret));
      if(!found||ret!=OB_SUCCESS)usleep(100000);
    } catch(const std::exception &error) {const int ret=OB_ERR_UNEXPECTED;connection.rollback();LOG_WARN("AI column worker exception","reason",error.what());usleep(100000);}
  }
}
}}
