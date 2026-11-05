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
#define USING_LOG_PREFIX SQL_RESV
#include "ai_column_definition.h"
#include "sql/resolver/ob_resolver_utils.h"
#include "sql/resolver/expr/ob_raw_expr_util.h"
#include "sql/printer/ob_raw_expr_printer.h"
#include "sql/session/ob_sql_session_info.h"
#include "share/schema/ob_table_schema.h"
#include "query/vector/ob_vector_index_util.h"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include <algorithm>
#include <set>
#include <stdexcept>

namespace oceanbase { namespace sql { namespace ai_column {
using namespace common;
using namespace share::schema;
namespace {
std::string parse_node_text(const ParseNode *node)
{ return node && node->str_value_ ? std::string(node->str_value_,node->str_len_) : std::string(); }
ObString definition_string(const std::string &s) { return ObString(s.size(),s.data()); }
void definition_require(bool valid) { if (!valid) throw std::invalid_argument("invalid AI COLUMN definition"); }
std::string definition_lower(std::string value)
{ for(char &c:value)if(c>='A'&&c<='Z')c+='a'-'A';return value; }
std::string definition_member(const rapidjson::Value &d,const char *name)
{ definition_require(d.HasMember(name)&&d[name].IsString());return std::string(d[name].GetString(),d[name].GetStringLength()); }
int print_definition_expr(ObResolverParams &params, ObRawExpr *expr,std::string &out)
{
  int ret=OB_SUCCESS;std::vector<char> buffer(OB_MAX_DEFAULT_VALUE_LENGTH);int64_t pos=0;
  ObRawExprPrinter printer(buffer.data(),buffer.size(),&pos,
      params.schema_checker_->get_schema_guard(),params.session_info_->get_timezone_info());
  if(OB_FAIL(printer.do_print(expr,T_NONE_SCOPE,true))) {} else out.assign(buffer.data(),pos);
  return ret;
}
int build_definition_expr(ObResolverParams &params,const ObTableSchema &table,const ObString &expression,
          ObRawExpr *&expr,ObIArray<ObQualifiedName> &columns,bool formalize=true)
{
  int ret=OB_SUCCESS;const ParseNode *node=nullptr;
  if(OB_FAIL(ObRawExprUtils::parse_expr_node_from_str(expression,
       params.session_info_->get_charsets4parser(),*params.allocator_,node))) {}
  else if(OB_FAIL(ObRawExprUtils::build_generated_column_expr(*params.expr_factory_,
       *params.session_info_,*node,expr,columns,&table,nullptr,params.schema_checker_,
       ObResolverUtils::DISABLE_CHECK))) {}
  for(int64_t i=0;OB_SUCC(ret)&&i<columns.count();++i) {
    auto &ref=columns.at(i);const auto *column=table.get_column_schema(ref.col_name_);
    if(ref.is_sys_func() || !ref.database_name_.empty() || !ref.tbl_name_.empty()
       || !column || column->is_hidden())ret=OB_ERR_BAD_FIELD_ERROR;
    else if(column->is_ai_column() || column->is_generated_column()) {
      ret=OB_NOT_SUPPORTED;
    } else if(OB_FAIL(ObRawExprUtils::init_column_expr(*column,params.session_info_,*ref.ref_expr_))) {}
  }
  if(OB_SUCC(ret)&&formalize)ret=expr->formalize(params.session_info_);
  return ret;
}
}
std::string quote_identifier(const std::string &name)
{ std::string s="`";for(char c:name){s+=c;if(c=='`')s+='`';}return s+'`'; }
std::string sql_string(const std::string &value)
{
  static const char hex[]="0123456789abcdef";std::string s="CONVERT(X'";
  for(unsigned char c:value){s+=hex[c>>4];s+=hex[c&15];}return s+"' USING utf8mb4)";
}
std::string Definition::encode() const
{
  rapidjson::Document d;d.SetObject();auto &a=d.GetAllocator();
  d.AddMember("version",version,a);d.AddMember("max_retries",max_retries,a);
  d.AddMember("dimension",dimension,a);d.AddMember("embedding",embedding,a);d.AddMember("input_is_json",input_is_json,a);
  auto add=[&](const char *key,const std::string &value){
    d.AddMember(rapidjson::Value(key,a),rapidjson::Value(value.data(),value.size(),a),a);
  };
  add("expression",expression);add("normalized_expr",normalized_expr);add("model",model);
  add("input_sql",input_sql);add("config_sql",config_sql);add("result_hint",result_hint);
  add("state_column",state_column);add("request_column",request_column);add("result_spec_column",result_spec_column);
  rapidjson::Value deps(rapidjson::kArrayType);
  for(const auto &dep:dependencies)deps.PushBack(rapidjson::Value(dep.data(),dep.size(),a),a);
  d.AddMember("dependencies",deps,a);rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);d.Accept(writer);
  return std::string(buffer.GetString(),buffer.GetSize());
}
Definition Definition::decode(const ObString &json)
{
  rapidjson::Document d;d.Parse(json.ptr(),json.length());
  definition_require(!d.HasParseError()&&d.IsObject()&&d.HasMember("version")&&d["version"].IsInt()&&d["version"].GetInt()==1);
  Definition out;definition_require(d.HasMember("max_retries")&&d["max_retries"].IsInt());
  out.max_retries=d["max_retries"].GetInt();definition_require(out.max_retries>=0&&out.max_retries<=100);
  definition_require(d.HasMember("dimension")&&d["dimension"].IsInt64()&&d.HasMember("embedding")&&d["embedding"].IsBool());
  out.dimension=d["dimension"].GetInt64();out.embedding=d["embedding"].GetBool();
  if(d.HasMember("input_is_json")){definition_require(d["input_is_json"].IsBool());out.input_is_json=d["input_is_json"].GetBool();}
  out.expression=definition_member(d,"expression");out.normalized_expr=definition_member(d,"normalized_expr");out.model=definition_member(d,"model");
  out.input_sql=definition_member(d,"input_sql");out.config_sql=definition_member(d,"config_sql");out.result_hint=definition_member(d,"result_hint");
  out.state_column=definition_member(d,"state_column");out.request_column=definition_member(d,"request_column");
  out.result_spec_column=definition_member(d,"result_spec_column");
  definition_require(d.HasMember("dependencies")&&d["dependencies"].IsArray());
  for(const auto &dep:d["dependencies"].GetArray()) {definition_require(dep.IsString());out.dependencies.emplace_back(dep.GetString(),dep.GetStringLength());}
  return out;
}
int capture(ObColumnSchemaV2 &column,const ParseNode &attribute)
{
  int ret=OB_SUCCESS;
  try {
    definition_require(!column.is_ai_column()&&attribute.num_child_==2&&attribute.children_[0]);
    Definition d;d.expression=parse_node_text(attribute.children_[0]);definition_require(!d.expression.empty());
    const ParseNode *options=attribute.children_[1];std::set<std::string> seen;
    for(int64_t i=0;options&&i<options->num_child_;++i) {
      const ParseNode *item=options->children_[i];definition_require(item&&item->type_==T_OP_EQ&&item->num_child_==2);
      const ParseNode *key=item->children_[0],*value=item->children_[1];
      definition_require(key&&value);if(key->type_==T_COLUMN_REF&&key->num_child_==3)key=key->children_[2];
      std::string name=definition_lower(parse_node_text(key));definition_require(seen.insert(name).second);
      if(name=="sync_mode")definition_require(value->type_==T_VARCHAR&&definition_lower(parse_node_text(value))=="async");
      else if(name=="max_retries") {definition_require(value->type_==T_INT&&value->value_>=0&&value->value_<=100);d.max_retries=value->value_;}
      else definition_require(false);
    }
    const std::string encoded=d.encode();ret=column.set_ai_column_info(definition_string(encoded));
  } catch(const std::bad_alloc &) {ret=OB_ALLOCATE_MEMORY_FAILED;}
    catch(const std::exception &) {ret=OB_INVALID_ARGUMENT;LOG_USER_ERROR(OB_INVALID_ARGUMENT,"AI COLUMN options");}
  return ret;
}
int bind(ObResolverParams &params,ObTableSchema &table,ObColumnSchemaV2 &column)
{
  int ret=OB_SUCCESS;
  try {
    Definition d=Definition::decode(column.get_ai_column_info_str());
    ObSEArray<ObQualifiedName,8> columns;ObRawExpr *expr=nullptr;
    if(column.is_rowkey_column()||column.is_autoincrement()
       ||!column.is_nullable()||!column.get_cur_default_value().is_null()) {
      ret=OB_NOT_SUPPORTED;LOG_USER_ERROR(OB_NOT_SUPPORTED,"AI COLUMN constraints or non-MySQL mode");
    } else if(OB_FAIL(build_definition_expr(params,table,definition_string(d.expression),expr,columns,false))) {}
    else if((expr->get_expr_type()!=T_FUN_SYS_AI_COMPLETE&&expr->get_expr_type()!=T_FUN_SYS_AI_EMBED)
        ||expr->get_param_count()<2||expr->get_param_count()>3||!expr->get_param_expr(0)->is_const_raw_expr())ret=OB_INVALID_ARGUMENT;
    else {
      const ObObj &model=static_cast<ObConstRawExpr *>(expr->get_param_expr(0))->get_value();
      if(!model.is_string_type()||model.get_string().empty())ret=OB_INVALID_ARGUMENT;
      else {auto name=model.get_string();d.model.assign(name.ptr(),name.length());}
      d.embedding=expr->get_expr_type()==T_FUN_SYS_AI_EMBED;
      const ObAiModelSchema *model_schema=nullptr;
      if(OB_SUCC(ret)&&OB_FAIL(params.schema_checker_->get_schema_guard()->get_ai_model_schema(definition_string(d.model),model_schema))) {}
      if(OB_SUCC(ret)&&(!model_schema||model_schema->get_type()!=(d.embedding?share::EndpointType::DENSE_EMBEDDING:share::EndpointType::COMPLETION)))ret=OB_INVALID_ARGUMENT;
      if(OB_SUCC(ret)&&d.embedding) {
        if(!column.is_collection())ret=OB_INVALID_ARGUMENT;
        else if(OB_FAIL(share::ObVectorIndexUtil::get_vector_dim_from_extend_type_info(column.get_extended_type_info(),d.dimension))) {}
        else if(expr->get_param_count()==3) {
          ObRawExpr *dim=expr->get_param_expr(2);
          if (!dim->is_const_raw_expr()
              || !ob_is_integer_type(static_cast<ObConstRawExpr *>(dim)->get_value().get_type())) {
            ret = OB_ERR_INVALID_VECTOR_DIM;
          } else {
            const int64_t requested_dim = static_cast<ObConstRawExpr *>(dim)->get_value().get_int();
            if (requested_dim != d.dimension) {
              ret = OB_ERR_INVALID_VECTOR_DIM;
              LOG_USER_ERROR(OB_ERR_INVALID_VECTOR_DIM, static_cast<uint32_t>(d.dimension),
                             static_cast<uint32_t>(requested_dim));
            }
          }
        }
      } else if(OB_SUCC(ret)) {
        if(column.is_json())d.result_hint="Return exactly one valid JSON value, with no markdown or additional text.";
        else if(ob_is_integer_type(column.get_data_type()))d.result_hint="Return exactly one base-10 integer, with no additional text.";
        else if(ob_is_numeric_type(column.get_data_type()))d.result_hint="Return exactly one finite base-10 number, with no additional text.";
        else if(ob_is_string_type(column.get_data_type()))d.result_hint="Return only the requested text.";
        else ret=OB_NOT_SUPPORTED;
      }
      // Bind literal arguments before formalization can insert implicit casts.
      if (OB_SUCC(ret) && OB_FAIL(expr->formalize(params.session_info_))) {}
      if (OB_SUCC(ret)) d.input_is_json = expr->get_param_expr(1)->get_data_type() == ObJsonType;
      if(OB_SUCC(ret)&&OB_FAIL(ObRawExprUtils::check_is_valid_generated_col(expr->get_param_expr(1),*params.allocator_))) {}
      if(OB_SUCC(ret)&&OB_FAIL(print_definition_expr(params,expr->get_param_expr(1),d.input_sql))) {}
      if(OB_SUCC(ret)&&!d.embedding&&expr->get_param_count()==3) {
        // Config expressions are evaluated while enqueueing in the business
        // transaction, so they must be pure just like the input expression.
        if(OB_FAIL(ObRawExprUtils::check_is_valid_generated_col(expr->get_param_expr(2),*params.allocator_))) {}
        else ret=print_definition_expr(params,expr->get_param_expr(2),d.config_sql);
      }
      d.dependencies.clear();
      for(int64_t i=0;OB_SUCC(ret)&&i<columns.count();++i) {
        auto name=columns.at(i).col_name_;std::string dep(name.ptr(),name.length());
        if(std::find(d.dependencies.begin(),d.dependencies.end(),dep)==d.dependencies.end())d.dependencies.push_back(dep);
      }
      // Stable names survive column-ID remapping when a table is copied/forked.
      auto name=column.get_column_name_str();uint64_t hash=1469598103934665603ULL;
      for(int i=0;i<name.length();++i){hash^=static_cast<unsigned char>(name.ptr()[i]);hash*=1099511628211ULL;}
      const std::string suffix=std::to_string(hash);
      d.state_column="__ai_state_"+suffix;d.request_column="__ai_request_"+suffix;d.result_spec_column="__ai_spec_"+suffix;
      const auto *previous=table.get_column_schema(column.get_column_name_str());
      const bool modify=previous&&previous->is_ai_column()&&previous->get_column_id()==column.get_column_id();
      if(modify) {
        const auto old=Definition::decode(previous->get_ai_column_info_str());
        if(!old.state_column.empty()){d.state_column=old.state_column;d.request_column=old.request_column;d.result_spec_column=old.result_spec_column;}
      }
      for(const auto &hidden:{d.state_column,d.request_column,d.result_spec_column}) {
        const auto *existing=table.get_column_schema(definition_string(hidden));
        if(existing&&!(modify&&existing->is_hidden()))ret=OB_ERR_COLUMN_DUPLICATE;
      }
      for(const auto &dep:d.dependencies)if(column.get_column_name_str().case_compare(definition_string(dep))==0)ret=OB_NOT_SUPPORTED;
      d.normalized_expr=d.expression;
      if(!d.result_hint.empty())d.normalized_expr+=" /* "+d.result_hint+" */";
      if(OB_SUCC(ret)) {const auto encoded=d.encode();ret=column.set_ai_column_info(definition_string(encoded));}
    }
  } catch(const std::bad_alloc &) {ret=OB_ALLOCATE_MEMORY_FAILED;}
    catch(const std::exception &) {ret=OB_INVALID_ARGUMENT;}
  if (OB_FAIL(ret)) {
    LOG_WARN("failed to bind AI column definition", K(ret), "column", column.get_column_name_str());
  }
  return ret;
}
int hidden_columns(const ObColumnSchemaV2 &column,std::vector<std::unique_ptr<ObColumnSchemaV2>> &columns)
{
  int ret=OB_SUCCESS;
  try {
    Definition d=Definition::decode(column.get_ai_column_info_str());columns.clear();
    for(int i=0;OB_SUCC(ret)&&i<3;++i) {
      columns.emplace_back(new ObColumnSchemaV2());auto &hidden=*columns.back();hidden.set_is_hidden(true);hidden.set_nullable(true);
      hidden.set_data_type(i==0?ObTinyIntType:ObVarcharType);
      hidden.set_collation_type(i==0?CS_TYPE_BINARY:CS_TYPE_UTF8MB4_BIN);
      hidden.set_charset_type(i==0?CHARSET_BINARY:CHARSET_UTF8MB4);
      if(i)hidden.set_data_length(i==1?64:256);
      const std::string &name=i==0?d.state_column:(i==1?d.request_column:d.result_spec_column);
      ret=hidden.set_column_name(definition_string(name));
      if(OB_SUCC(ret)&&i==0) {ObObj value;value.set_tinyint(0);ret=hidden.set_cur_default_value(value,false);if(OB_SUCC(ret))ret=hidden.set_orig_default_value(value);}
    }
  } catch(const std::bad_alloc &) {ret=OB_ALLOCATE_MEMORY_FAILED;}
    catch(const std::exception &) {ret=OB_INVALID_ARGUMENT;}
  return ret;
}
int resolve_input(ObResolverParams &params,const ObTableSchema &table,const Definition &d,ObRawExpr *&input)
{ ObSEArray<ObQualifiedName,8> columns;return build_definition_expr(params,table,definition_string(d.input_sql),input,columns); }
}}} // oceanbase::sql::ai_column
