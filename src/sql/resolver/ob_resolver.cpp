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
#include "sql/resolver/ob_resolver.h"
#include "sql/parser/ob_parser.h"
#include "sql/engine/basic/file_scan_utils.h"
#include "sql/ai_column/ai_column_runtime.h"
#include <cctype>
#include <set>
#include "lib/utility/ob_smart_call.h"
#include "lib/stat/ob_diagnostic_info_guard.h"
#include "sql/resolver/cmd/ob_alter_system_resolver.h"
#include "sql/resolver/dml/ob_insert_resolver.h"
#include "sql/resolver/dml/ob_update_resolver.h"
#include "sql/resolver/dml/ob_delete_resolver.h"
#include "sql/resolver/tcl/ob_start_trans_resolver.h"
#include "sql/resolver/tcl/ob_end_trans_resolver.h"
#include "sql/resolver/prepare/ob_prepare_resolver.h"
#include "sql/resolver/prepare/ob_execute_resolver.h"
#include "sql/resolver/prepare/ob_deallocate_resolver.h"
#include "sql/resolver/ddl/ob_create_table_resolver.h"
#include "sql/resolver/ddl/ob_rename_table_resolver.h"
#include "sql/resolver/ddl/ob_truncate_table_resolver.h"
#include "sql/resolver/ddl/ob_fork_table_resolver.h"
#include "sql/resolver/ddl/ob_fork_database_resolver.h"
#include "sql/resolver/ddl/ob_create_table_like_resolver.h"
#include "sql/resolver/ddl/ob_alter_table_resolver.h"
#include "sql/resolver/ddl/ob_drop_table_resolver.h"
#include "sql/resolver/ddl/ob_create_index_resolver.h"
#include "sql/resolver/ddl/ob_drop_index_resolver.h"
#include "sql/resolver/ddl/ob_create_database_resolver.h"
#include "sql/resolver/ddl/ob_alter_database_resolver.h"
#include "sql/resolver/ddl/ob_use_database_resolver.h"
#include "sql/resolver/ddl/ob_drop_database_resolver.h"
#include "sql/resolver/ddl/ob_create_view_resolver.h"
#include "sql/resolver/ddl/ob_explain_resolver.h"
#include "sql/resolver/ddl/ob_create_outline_resolver.h"
#include "sql/resolver/ddl/ob_alter_outline_resolver.h"
#include "sql/resolver/ddl/ob_drop_outline_resolver.h"
#include "sql/resolver/ddl/ob_drop_routine_resolver.h"
#include "sql/resolver/ddl/ob_trigger_resolver.h"
#include "sql/resolver/ddl/ob_optimize_resolver.h"
#include "ddl/ob_drop_routine_resolver.h"
#include "ddl/ob_alter_routine_resolver.h"
#include "sql/resolver/ddl/ob_create_package_resolver.h"
#include "sql/resolver/ddl/ob_drop_package_resolver.h"
#include "sql/resolver/ddl/ob_recyclebin_restore_resolver.h"
#include "sql/resolver/ddl/ob_purge_resolver.h"
#include "sql/resolver/ddl/ob_analyze_stmt_resolver.h"
#include "sql/resolver/ddl/ob_purge_resolver.h"
#include "sql/resolver/ddl/ob_set_comment_resolver.h"
#include "sql/resolver/ddl/ob_lock_table_resolver.h"
#include "sql/resolver/dml/ob_insert_resolver.h"
#include "sql/resolver/dml/ob_update_resolver.h"
#include "sql/resolver/dml/ob_delete_resolver.h"
#include "sql/resolver/dcl/ob_create_user_resolver.h"
#include "sql/resolver/dcl/ob_drop_user_resolver.h"
#include "sql/resolver/dcl/ob_rename_user_resolver.h"
#include "sql/resolver/dcl/ob_set_password_resolver.h"
#include "sql/resolver/dcl/ob_lock_user_resolver.h"
#include "sql/resolver/dcl/ob_grant_resolver.h"
#include "sql/resolver/dcl/ob_revoke_resolver.h"
#include "sql/resolver/dcl/ob_create_role_resolver.h"
#include "sql/resolver/dcl/ob_drop_role_resolver.h"
#include "sql/resolver/dcl/ob_alter_user_role_resolver.h"
#include "sql/resolver/tcl/ob_start_trans_resolver.h"
#include "sql/resolver/tcl/ob_end_trans_resolver.h"
#include "tcl/ob_savepoint_resolver.h"
#include "sql/resolver/cmd/ob_variable_set_resolver.h"
#include "sql/resolver/cmd/ob_show_resolver.h"
#include "sql/resolver/cmd/ob_diff_table_resolver.h"
#include "sql/resolver/cmd/ob_merge_table_resolver.h"
#include "sql/resolver/cmd/ob_kill_resolver.h"
#include "sql/resolver/cmd/ob_set_names_resolver.h"
#include "sql/resolver/cmd/ob_set_transaction_resolver.h"
#include "sql/resolver/cmd/ob_empty_query_resolver.h"
#include "sql/resolver/cmd/ob_anonymous_block_resolver.h"
#include "sql/resolver/cmd/ob_call_procedure_resolver.h"
#include "sql/resolver/cmd/ob_load_data_resolver.h"
#include "sql/resolver/prepare/ob_execute_resolver.h"
#include "sql/resolver/prepare/ob_deallocate_resolver.h"
#include "sql/resolver/ddl/ob_purge_resolver.h"
#include "sql/resolver/ddl/ob_set_comment_resolver.h"
#include "sql/resolver/expr/ob_raw_expr_wrap_enum_set.h"
#include "sql/resolver/cmd/ob_get_diagnostics_resolver.h"
#include "sql/resolver/dcl/ob_alter_role_resolver.h"
#include "sql/pl/ob_pl_package.h"

namespace oceanbase
{
using namespace common;
using namespace share;
namespace sql
{
namespace {
bool has_file_table(const ParseNode &node, int depth = 0)
{
  if (node.type_ == T_FILE_TABLE) return true;
  if (depth >= 256) return false;
  for (int64_t i = 0; i < node.num_child_; ++i) {
    if (OB_NOT_NULL(node.children_[i]) && has_file_table(*node.children_[i],depth+1)) return true;
  }
  return false;
}
}

ObResolver::ObResolver(ObResolverParams &params)
    : params_(params)
{

}

ObResolver::~ObResolver()
{
}

template <typename ResolverType>
int ObResolver::stmt_resolver_func(ObResolverParams &params, const ParseNode &parse_tree, ObStmt *&stmt)
{
  int ret = OB_SUCCESS;
  HEAP_VAR(ResolverType, stmt_resolver, params) {
    if (OB_FAIL(SMART_CALL(stmt_resolver.resolve(parse_tree)))) {
    }
    stmt = stmt_resolver.get_basic_stmt();
  }
  return ret;
}

template <typename SelectResolverType>
int ObResolver::select_stmt_resolver_func(ObResolverParams &params, const ParseNode &parse_tree, ObStmt *&stmt)
{
  int ret = OB_SUCCESS;
  HEAP_VAR(SelectResolverType, stmt_resolver, params) {
    stmt_resolver.set_calc_found_rows(true);
    stmt_resolver.set_has_top_limit(true);
    if (OB_FAIL(SMART_CALL(stmt_resolver.resolve(parse_tree)))) {
    }
    stmt = stmt_resolver.get_basic_stmt();
  }
  return ret;
}

int ObResolver::resolve_file_copy(const ParseNode &copy, ObStmt *&stmt)
{
  int ret = OB_SUCCESS;
  bool enabled = false;
  if (copy.num_child_ != 3 || OB_ISNULL(copy.children_[0]) || OB_ISNULL(copy.children_[1])) {
    ret = OB_ERR_UNEXPECTED;
  } else if (OB_FAIL(params_.session_info_->get_sys_variable(share::SYS_VAR_ENABLE_FILE_SQL,enabled))) {
  } else if (!enabled) {
    ret = OB_NOT_SUPPORTED; LOG_USER_ERROR(OB_NOT_SUPPORTED,"COPY requires enable_file_sql = ON");
  } else {
    const_cast<ObGlobalHint &>(params_.query_ctx_->query_hint_.get_global_hint()).plan_cache_policy_ = OB_USE_PLAN_CACHE_NONE;
    const ParseNode *select=copy.children_[0];
    ParseResult parsed;
    ObSqlString table_sql;
    if (select->type_ != T_SELECT) {
      if (select->type_ != T_RELATION_FACTOR || select->num_child_<2 || OB_ISNULL(select->children_[1])) {
        ret = OB_ERR_PARSE_SQL;
      } else if (OB_FAIL(table_sql.append("SELECT * FROM "))) {
      } else {
        for (int i=0; OB_SUCC(ret) && i<2; ++i) {
          const ParseNode *part=select->children_[i];if(OB_ISNULL(part))continue;
          if (OB_FAIL(table_sql.append("`"))) {
          }
          for (int64_t j=0;OB_SUCC(ret) && j<part->str_len_;++j) {
            if (part->str_value_[j]=='`' && OB_FAIL(table_sql.append("`"))) {
            } else if (OB_SUCC(ret)) ret=table_sql.append(part->str_value_+j,1);
          }
          if (OB_SUCC(ret)) ret=table_sql.append(i==0?"`.":"`");
        }
        if (OB_SUCC(ret)) {
          ObParser parser(*params_.allocator_,params_.session_info_->get_sql_mode());
          if (OB_FAIL(parser.parse(table_sql.string(),parsed))) {
          } else if (OB_ISNULL(parsed.result_tree_) || parsed.result_tree_->num_child_ != 1) ret=OB_ERR_UNEXPECTED;
          else select=parsed.result_tree_->children_[0];
        }
      }
    }
    ObResolver child(params_);
    if (OB_FAIL(ret)) {
    } else if (OB_FAIL(child.resolve(IS_NOT_PREPARED_STMT,*select,stmt))) {
    } else if (OB_ISNULL(stmt) || !stmt->is_select_stmt()) {
      ret=OB_ERR_UNEXPECTED;
    } else {
      ObSelectStmt &query=*static_cast<ObSelectStmt *>(stmt);
      if (query.has_select_into()) return OB_INVALID_ARGUMENT;
      try {
        file_sql::Definition definition;
        definition.path.assign(copy.children_[1]->str_value_,copy.children_[1]->str_len_);
        std::set<std::string> seen;
        const ParseNode *options=copy.children_[2];
        for (int64_t i=0;OB_SUCC(ret) && OB_NOT_NULL(options) && i<options->num_child_;++i) {
          const ParseNode *option=options->children_[i];
          if (OB_ISNULL(option) || option->num_child_ != 2 || OB_ISNULL(option->children_[0]) || OB_ISNULL(option->children_[1])) {
            ret=OB_INVALID_ARGUMENT;break;
          }
          std::string name(option->children_[0]->str_value_,option->children_[0]->str_len_);
          for(char &c:name)if(c>='A' && c<='Z')c+='a'-'A';
          if(!seen.insert(name).second) {ret=OB_INVALID_ARGUMENT;break;}
          const ParseNode *value=option->children_[1];
          if(name=="format") {
            ObString format(value->str_len_,value->str_value_);
            if(format.case_compare("csv")!=0)ret=OB_NOT_SUPPORTED;
          } else if(name=="header") {
            if((value->type_!=T_BOOL && value->type_!=T_INT) || value->value_<0 || value->value_>1)ret=OB_INVALID_ARGUMENT;
            else definition.options.header=value->value_!=0;
          } else if(value->type_==T_VARCHAR) {
            if(name=="nullstr" || name=="null")definition.options.nullstr.assign(value->str_value_,value->str_len_);
            else if(value->str_len_!=1)ret=OB_INVALID_ARGUMENT;
            else if(name=="delimiter" || name=="delim" || name=="sep")definition.options.delimiter=value->str_value_[0];
            else if(name=="quote")definition.options.quote=value->str_value_[0];
            else if(name=="escape")definition.options.escape=value->str_value_[0];
            else ret=OB_INVALID_ARGUMENT;
          } else ret=OB_INVALID_ARGUMENT;
        }
        for(int64_t i=0;OB_SUCC(ret) && i<query.get_select_item_size();++i) {
          const ObString &name=query.get_select_item(i).alias_name_;
          file_sql::Column column;column.name.assign(name.ptr(),name.length());column.type=file_sql::Type::VARCHAR;
          definition.columns.push_back(std::move(column));
        }
        if(OB_SUCC(ret)) {
          std::string encoded=file_sql::FileReader::encode(definition);
          ObString descriptor(encoded.size(),encoded.data());
          ObSelectIntoItem *into=OB_NEWx(ObSelectIntoItem,params_.allocator_);
          if(OB_ISNULL(into))ret=OB_ALLOCATE_MEMORY_FAILED;
          else if(OB_FAIL(ob_write_string(*params_.allocator_,descriptor,into->external_properties_))) {
          } else {
            ObString path;
            ObString input(definition.path.size(),definition.path.data());
            if(OB_FAIL(ob_write_string(*params_.allocator_,input,path))) {
            } else {
              into->into_type_=T_COPY_FILE;into->outfile_name_.set_varchar(path);
              into->outfile_name_.set_collation_type(CS_TYPE_UTF8MB4_BIN);query.set_select_into(into);
            }
          }
        }
      } catch(const std::bad_alloc &) {ret=OB_ALLOCATE_MEMORY_FAILED;}
    }
  }
  return ret;
}

namespace {
// Rewrite only bare identifiers. Strings, quoted identifiers, and comments retain
// their SQL meaning, including a backtick-quoted business column named status.
std::string retry_condition(const std::string &sql,const std::string &state,bool backslash)
{
  const std::string pseudo="(CASE COALESCE("+state+",0) WHEN 0 THEN 'PENDING' WHEN 1 THEN 'READY' ELSE 'FAILED' END)";
  std::string out;size_t i=0;
  auto ident=[](unsigned char c){return std::isalnum(c)||c=='_'||c=='$'||c>=128;};
  while(i<sql.size()) {
    size_t begin=i;char c=sql[i];
    if(c=='\''||c=='"'||c=='`') {
      ++i;while(i<sql.size()) {char next=sql[i++];
        if(backslash&&next=='\\'&&c!='`'&&i<sql.size())++i;
        else if(next==c){if(i<sql.size()&&sql[i]==c)++i;else break;}}
      out.append(sql,begin,i-begin);
    } else if(c=='#'||(c=='-'&&i+2<sql.size()&&sql[i+1]=='-'&&std::isspace(static_cast<unsigned char>(sql[i+2])))) {
      while(i<sql.size()&&sql[i]!='\n')++i;out.append(sql,begin,i-begin);
    } else if(c=='/'&&i+1<sql.size()&&sql[i+1]=='*') {
      i+=2;while(i+1<sql.size()&&!(sql[i]=='*'&&sql[i+1]=='/'))++i;
      if(i+1<sql.size())i+=2;out.append(sql,begin,i-begin);
    } else if(ident(static_cast<unsigned char>(c))) {
      while(i<sql.size()&&ident(static_cast<unsigned char>(sql[i])))++i;
      std::string word=sql.substr(begin,i-begin);for(char &ch:word)if(ch>='A'&&ch<='Z')ch+='a'-'A';
      if(word=="status"&&(begin==0||(sql[begin-1]!='.'&&sql[begin-1]!='@')))out+=pseudo;
      else out.append(sql,begin,i-begin);
    } else out+=sql[i++];
  }
  return out;
}
}
int ObResolver::resolve_ai_column_statement(const ParseNode &node,ObStmt *&stmt)
{
  int ret=OB_SUCCESS;
  try {
    const bool retry=node.type_==T_AI_COLUMN_RETRY;
    if(node.num_child_!=(retry?4:1)||!node.children_[0])return OB_ERR_UNEXPECTED;
    if(retry&&node.children_[3]){LOG_USER_ERROR(OB_NOT_SUPPORTED,"combining AI COLUMN RETRY with another ALTER action");return OB_NOT_SUPPORTED;}
    const ParseNode *factor=node.children_[0];
    if(factor->type_!=T_RELATION_FACTOR||factor->num_child_<2||!factor->children_[1])return OB_ERR_UNEXPECTED;
    ObString database=params_.session_info_->get_database_name();
    if(factor->children_[0])database.assign_ptr(factor->children_[0]->str_value_,factor->children_[0]->str_len_);
    ObString name(factor->children_[1]->str_len_,factor->children_[1]->str_value_);
    const share::schema::ObTableSchema *table=nullptr;
    if(OB_FAIL(params_.schema_checker_->get_table_schema(database,name,false,table)))return ret;
    if(!table)return OB_TABLE_NOT_EXIST;
    std::string sql;
    if(retry) {
      const ParseNode *column_node=node.children_[1];if(!column_node)return OB_ERR_UNEXPECTED;
      if(column_node->type_==T_COLUMN_REF&&column_node->num_child_==3)column_node=column_node->children_[2];
      if(!column_node)return OB_ERR_UNEXPECTED;
      const auto *column=table->get_column_schema(ObString(column_node->str_len_,column_node->str_value_));
      if(!column||!column->is_ai_column()){LOG_USER_ERROR(OB_NOT_SUPPORTED,"retrying a column that is not an AI column");return OB_NOT_SUPPORTED;}
      const auto d=ai_column::Definition::decode(column->get_ai_column_info_str());
      const auto target=ai_column::quote_identifier(std::string(database.ptr(),database.length()))+"."+ai_column::quote_identifier(std::string(name.ptr(),name.length()));
      sql="UPDATE "+target+" SET "+ai_column::quote_identifier(d.result_spec_column)+"=NULL,"+
          ai_column::quote_identifier(std::string(column->get_column_name_str().ptr(),column->get_column_name_str().length()))+"=NULL,"+
          ai_column::quote_identifier(d.state_column)+"=0";
      if(node.children_[2]) {
        const ParseNode *condition=node.children_[2];
        bool no_backslash=false;IS_NO_BACKSLASH_ESCAPES(params_.session_info_->get_sql_mode(),no_backslash);
        sql+=" WHERE "+retry_condition(std::string(condition->str_value_,condition->str_len_),ai_column::quote_identifier(d.state_column),!no_backslash);
      }
    } else {
      sql="SELECT column_name,normalized_expr,sync_mode,max_retries,definition_version,state_column_id,request_column_id,result_spec_column_id,dependencies FROM oceanbase.__all_ai_column WHERE table_id="+
          std::to_string(table->get_table_id())+" ORDER BY column_id";
    }
    ParseResult parsed;ObParser parser(*params_.allocator_,params_.session_info_->get_sql_mode());
    if(OB_FAIL(parser.parse(ObString(sql.size(),sql.data()),parsed)))return ret;
    if(!parsed.result_tree_||parsed.result_tree_->num_child_!=1||!parsed.result_tree_->children_[0])return OB_ERR_UNEXPECTED;
    ai_column::InternalWriteGuard trusted(*params_.session_info_);
    const_cast<ObGlobalHint &>(params_.query_ctx_->query_hint_.get_global_hint()).plan_cache_policy_=OB_USE_PLAN_CACHE_NONE;
    ObResolver resolver(params_);ret=resolver.resolve(IS_NOT_PREPARED_STMT,*parsed.result_tree_->children_[0],stmt);
  } catch(const std::bad_alloc &){ret=OB_ALLOCATE_MEMORY_FAILED;}
    catch(const std::exception &){ret=OB_INVALID_ARGUMENT;}
  return ret;
}

int ObResolver::resolve_file_description(const ParseNode &select, ObStmt *&stmt)
{
  int ret = OB_SUCCESS;
  ObStmt *resolved = nullptr;
  ObResolver child(params_);
  if (OB_FAIL(child.resolve(IS_NOT_PREPARED_STMT,select,resolved))) {
  } else if (OB_ISNULL(resolved) || !resolved->is_select_stmt()) {
    ret = OB_ERR_UNEXPECTED;
  } else {
    const ObSelectStmt &query = *static_cast<ObSelectStmt *>(resolved);
    ObSqlString sql;
    if (OB_FAIL(sql.append("SELECT column_name,column_type,`null`,`key`,`default`,extra FROM ("))) {
    }
    for (int64_t i = 0; OB_SUCC(ret) && i < query.get_select_item_size(); ++i) {
      const SelectItem &item = query.get_select_item(i);
      const ObString &name = item.alias_name_;
      const char *type = "VARCHAR";
      switch (item.expr_->get_data_type()) {
        case ObNullType: type = "NULL"; break;
        case ObIntType: case ObUInt64Type: type = "BIGINT"; break;
        case ObDoubleType: case ObFloatType: case ObNumberType: type = "DOUBLE"; break;
        case ObTinyIntType: type = "BOOLEAN"; break;
        case ObDateType: type = "DATE"; break;
        case ObDateTimeType: case ObTimestampType: type = "DATETIME"; break;
        default: break;
      }
      if (i > 0 && OB_FAIL(sql.append(" UNION ALL "))) {
      } else if (OB_FAIL(sql.append("SELECT CONVERT(X'"))) {
      }
      for (int64_t j = 0; OB_SUCC(ret) && j < name.length(); ++j) {
        ret = sql.append_fmt("%02x",static_cast<unsigned char>(name.ptr()[j]));
      }
      if (OB_SUCC(ret)) {
        ret = sql.append_fmt("' USING utf8mb4) AS column_name,'%s' AS column_type,'%s' AS `null`,"
                             "NULL AS `key`,NULL AS `default`,NULL AS extra,%ld AS ordinal",
                             type,(item.expr_->get_result_flag() & NOT_NULL_FLAG) ? "NO" : "YES",i);
      }
    }
    if (OB_SUCC(ret) && OB_FAIL(sql.append(") file_columns ORDER BY ordinal"))) {
    } else if (OB_SUCC(ret)) {
      ParseResult parsed;
      ObParser parser(*params_.allocator_,params_.session_info_->get_sql_mode());
      if (OB_FAIL(parser.parse(sql.string(),parsed))) {
      } else if (OB_ISNULL(parsed.result_tree_) || parsed.result_tree_->num_child_ != 1
                 || OB_ISNULL(parsed.result_tree_->children_[0])) {
        ret = OB_ERR_UNEXPECTED;
      } else {
        ObResolver result(params_);
        ret = result.resolve(IS_NOT_PREPARED_STMT,*parsed.result_tree_->children_[0],stmt);
      }
    }
  }
  return ret;
}

int ObResolver::resolve(IsPrepared if_prepared, const ParseNode &parse_tree, ObStmt *&stmt)
{
#define REGISTER_STMT_RESOLVER(name)                                   \
  do {                                                                 \
    ret = stmt_resolver_func<Ob##name##Resolver>(params_, *real_parse_tree, stmt); \
  } while (0)

#define REGISTER_SELECT_STMT_RESOLVER(name)                                   \
  do {                                                                        \
    ret = select_stmt_resolver_func<Ob##name##Resolver>(params_, parse_tree, stmt); \
  } while (0)
  ACTIVE_SESSION_FLAG_SETTER_GUARD(in_resolve);
  int ret = OB_SUCCESS;
  const ParseNode *real_parse_tree = NULL;
  int64_t questionmark_count = 0;
  UNUSED(if_prepared);
  if (OB_ISNULL(params_.allocator_)
      || OB_ISNULL(params_.schema_checker_)
      || OB_ISNULL(params_.session_info_)
      || OB_ISNULL(params_.expr_factory_)
      || OB_ISNULL(params_.query_ctx_)) {
    ret = OB_NOT_INIT;
  } else {
    real_parse_tree = &parse_tree;
  }

  if (OB_SUCC(ret)) {
    params_.resolver_scope_stmt_type_ = parse_tree.type_;
    params_.session_info_->set_stmt_type(
      ObResolverUtils::get_stmt_type_by_item_type(parse_tree.type_));
    ObExecContext *exec_ctx = params_.session_info_->get_cur_exec_ctx();
    if (exec_ctx != nullptr && exec_ctx->get_sql_ctx() != nullptr) {
      //save the current stmt type on my context,
      //some other modules may need to get the current SQL stmt type through its context
      ObSqlCtx *sql_ctx = exec_ctx->get_sql_ctx();
      sql_ctx->stmt_type_ = ObResolverUtils::get_stmt_type_by_item_type(parse_tree.type_);
    }

    switch (real_parse_tree->type_) {
      case T_CREATE_TABLE: {
        REGISTER_STMT_RESOLVER(CreateTable);
        break;
      }
      case T_ALTER_TABLE: {
        REGISTER_STMT_RESOLVER(AlterTable);
        break;
      }
      case T_CREATE_INDEX: {
        REGISTER_STMT_RESOLVER(CreateIndex);
        break;
      }
      case T_CREATE_VIEW: {
        REGISTER_STMT_RESOLVER(CreateView);
        break;
      }
      case T_ALTER_VIEW: {
#if 0
        ObAlterViewResolver stmt_resolver(params_);
        if (OB_FAIL(stmt_resolver.resolve(*node))) {
        }

        stmt = stmt_resolver.get_basic_stmt();
#endif
        ret = OB_NOT_SUPPORTED;
        LOG_WARN("not support to alter view");
	      LOG_USER_ERROR(OB_NOT_SUPPORTED, "alter view");
        break;
      }
      case T_DROP_VIEW:
      case T_DROP_TABLE: {
        REGISTER_STMT_RESOLVER(DropTable);
        break;
      }
      case T_DROP_INDEX: {
        REGISTER_STMT_RESOLVER(DropIndex);
        break;
      }
      case T_CREATE_TABLE_LIKE: {
        REGISTER_STMT_RESOLVER(CreateTableLike);
        break;
      }
      case T_FORK_TABLE: {
        REGISTER_STMT_RESOLVER(ForkTable);
        break;
      }
      case T_FORK_DATABASE: {
        REGISTER_STMT_RESOLVER(ForkDatabase);
        break;
      }
      case T_DIFF_TABLE: {
        REGISTER_STMT_RESOLVER(DiffTable);
        break;
      }
      case T_MERGE_TABLE: {
        REGISTER_STMT_RESOLVER(MergeTable);
        break;
      }
      case T_SELECT: {
        REGISTER_SELECT_STMT_RESOLVER(Select);
        break;
      }
      case T_INSERT: {
        REGISTER_STMT_RESOLVER(Insert);
        break;
      }
      case T_UPDATE: {
        REGISTER_STMT_RESOLVER(Update);
        break;
      }
      case T_DELETE: {
        REGISTER_STMT_RESOLVER(Delete);
        break;
      }
      case T_BEGIN: {
        REGISTER_STMT_RESOLVER(StartTrans);
        break;
      }
      case T_ROLLBACK:
        //fall through
      case T_COMMIT: {
        REGISTER_STMT_RESOLVER(EndTrans);
        break;
      }
      case T_FREEZE: {
        REGISTER_STMT_RESOLVER(Freeze);
        break;
      }
      case T_FLUSH_CACHE: {
        REGISTER_STMT_RESOLVER(FlushCache);
        break;
      }
      case T_FLUSH_KVCACHE: {
        REGISTER_STMT_RESOLVER(FlushKVCache);
        break;
      }
      case T_FLUSH_ILOGCACHE: {
        REGISTER_STMT_RESOLVER(FlushIlogCache);
        break;
      }
      case T_FLUSH_DAG_WARNINGS: {
        REGISTER_STMT_RESOLVER(FlushDagWarnings);
        break;
      }
      case T_MERGE_CONTROL: {
        REGISTER_STMT_RESOLVER(AdminMerge);
        break;
      }
      case T_CANCEL_TASK: {
        REGISTER_STMT_RESOLVER(CancelTask);
        break;
      }
      case T_REFRESH_MEMORY_STAT: {
        REGISTER_STMT_RESOLVER(RefreshMemStat);
        break;
      }
      case T_REFRESH_IO_CALIBRATION: {
        REGISTER_STMT_RESOLVER(RefreshIOCalibration);
        break;
      }
      case T_SWITCHOVER_TO_STANDBY:
      case T_SWITCHOVER_TO_PRIMARY:
      case T_ACTIVATE_STANDBY: {
        REGISTER_STMT_RESOLVER(SwitchRole);
        break;
      }
      case T_ALTER_SYSTEM_SET_PARAMETER: {
        REGISTER_STMT_RESOLVER(SetConfig);
        break;
      }
      case T_ALTER_SYSTEM_SETTP: {
        REGISTER_STMT_RESOLVER(SetTP);
        break;
      }
      case T_CLEAR_MERGE_ERROR: {
        REGISTER_STMT_RESOLVER(ClearMergeError);
        break;
      }
      case T_CREATE_DATABASE: {
        REGISTER_STMT_RESOLVER(CreateDatabase);
        break;
      }
      case T_USE_DATABASE: {
        REGISTER_STMT_RESOLVER(UseDatabase);
        break;
      }
      case T_ALTER_DATABASE: {
        REGISTER_STMT_RESOLVER(AlterDatabase);
        break;
      }
      case T_DROP_DATABASE: {
        REGISTER_STMT_RESOLVER(DropDatabase);
        break;
      }
      case T_RENAME_TABLE: {
        REGISTER_STMT_RESOLVER(RenameTable);
        break;
      }
      case T_TRUNCATE_TABLE: {
        REGISTER_STMT_RESOLVER(TruncateTable);
        break;
      }
      case T_RECYCLEBIN_RESTORE_TABLE: {
        REGISTER_STMT_RESOLVER(RecyclebinRestoreTable);
        break;
      }
      case T_RECYCLEBIN_RESTORE_DATABASE: {
        REGISTER_STMT_RESOLVER(RecyclebinRestoreDatabase);
        break;
      }
      case T_PURGE_TABLE: {
        REGISTER_STMT_RESOLVER(PurgeTable);
        break;
      }
      case T_PURGE_INDEX: {
        REGISTER_STMT_RESOLVER(PurgeIndex);
        break;
      }
      case T_PURGE_DATABASE: {
        REGISTER_STMT_RESOLVER(PurgeDatabase);
        break;
      }
      case T_PURGE_RECYCLEBIN: {
        REGISTER_STMT_RESOLVER(PurgeRecycleBin);
        break;
      }
      case T_OPTIMIZE_TABLE: {
        REGISTER_STMT_RESOLVER(OptimizeTable);
        break;
      }
      case T_PREPARE: {
        if (params_.is_prepare_protocol_) {
          ret = OB_ERR_PARSE_SQL;
        } else {
          REGISTER_STMT_RESOLVER(Prepare);
        }
        break;
      }
      case T_EXECUTE: {
        if (params_.is_prepare_protocol_) {
          ret = OB_ERR_PARSE_SQL;
        } else {
          REGISTER_STMT_RESOLVER(Execute);
        }
        break;
      }
      case T_DEALLOCATE: {
        if (params_.is_prepare_protocol_) {
          ret = OB_ERR_PARSE_SQL;
        } else {
          REGISTER_STMT_RESOLVER(Deallocate);
        }
        break;
      }
      case T_VARIABLE_SET: {
        REGISTER_STMT_RESOLVER(VariableSet);
        break;
      }
      case T_TRANSACTION: {
        REGISTER_STMT_RESOLVER(SetTransaction);
        break;
      }
      case T_SHOW_TABLES:
      case T_SHOW_DATABASES:
      case T_SHOW_VARIABLES:
      case T_SHOW_COLUMNS:
      case T_SHOW_SCHEMA:
      case T_SHOW_CREATE_DATABASE:
      case T_SHOW_CREATE_TABLE:
      case T_SHOW_CREATE_VIEW:
      case T_SHOW_CREATE_PROCEDURE:
      case T_SHOW_CREATE_FUNCTION:
      case T_SHOW_TABLE_STATUS:
      case T_SHOW_PARAMETERS:
      case T_SHOW_INDEXES:
      case T_SHOW_PROCESSLIST:
      case T_SHOW_WARNINGS:
      case T_SHOW_ERRORS:
      case T_SHOW_TRACE:
      case T_SHOW_ENGINES:
      case T_SHOW_PROFILE:
      case T_SHOW_ENGINE:
      case T_SHOW_OPEN_TABLES:
      case T_SHOW_PRIVILEGES:
      case T_SHOW_CHARSET:
      case T_SHOW_COLLATION:
      case T_SHOW_GRANTS:
      case T_SHOW_RECYCLEBIN:
      case T_SHOW_PROCEDURE_STATUS:
      case T_SHOW_FUNCTION_STATUS:
      case T_SHOW_TRIGGERS:
      case T_SHOW_STATUS:
      case T_SHOW_CREATE_TRIGGER:
      case T_SHOW_CHECK_TABLE:
      case T_SHOW_CREATE_USER: {
        REGISTER_STMT_RESOLVER(Show);
        break;
      }
      case T_CREATE_USER: {
        REGISTER_STMT_RESOLVER(CreateUser);
        break;
      }
      case T_DROP_USER: {
        REGISTER_STMT_RESOLVER(DropUser);
        break;
      }
      case T_RENAME_USER: {
        REGISTER_STMT_RESOLVER(RenameUser);
        break;
      }
      case T_SET_PASSWORD: {
        REGISTER_STMT_RESOLVER(SetPassword);
        break;
      }
      case T_LOCK_USER: {
        REGISTER_STMT_RESOLVER(LockUser);
        break;
      }
      case T_ALTER_USER_DEFAULT_ROLE:
      case T_SET_ROLE: {
        REGISTER_STMT_RESOLVER(AlterUserRole);
        break;
      }

      case T_GRANT_ROLE:
      case T_SYSTEM_GRANT:
      case T_GRANT: {
        REGISTER_STMT_RESOLVER(Grant);
        break;
      }
      case T_SYSTEM_REVOKE:
      case T_REVOKE:
      case T_REVOKE_ALL: {
        REGISTER_STMT_RESOLVER(Revoke);
        break;
      }
      case T_AI_COLUMN_RETRY:
      case T_SHOW_AI_COLUMNS:
        ret = resolve_ai_column_statement(parse_tree, stmt);
        break;
      case T_COPY_FILE: {
        ret = resolve_file_copy(parse_tree,stmt);
        break;
      }
      case T_EXPLAIN: {
        if (parse_tree.is_file_describe_ && parse_tree.num_child_ == 5
            && OB_NOT_NULL(parse_tree.children_[4]) && parse_tree.children_[4]->type_ == T_SELECT
            && has_file_table(*parse_tree.children_[4])) {
          ret = resolve_file_description(*parse_tree.children_[4],stmt);
        } else {
          REGISTER_STMT_RESOLVER(Explain);
        }
        break;
      }
      case T_ALTER_SYSTEM_SET: {
        REGISTER_STMT_RESOLVER(AlterSystemSet);
        break;
      }
      case T_ALTER_SYSTEM_KILL: {
        REGISTER_STMT_RESOLVER(Kill);
        break;
      }
      case T_ALTER_SESSION_SET: {
        REGISTER_STMT_RESOLVER(AlterSessionSet);
        break;
      }
      case T_SET_NAMES:
        // fall through
      case T_SET_CHARSET: {
        REGISTER_STMT_RESOLVER(SetNames);
        break;
      }
      case T_KILL: {
        REGISTER_STMT_RESOLVER(Kill);
        break;
      }
      case T_EMPTY_QUERY: {
        REGISTER_STMT_RESOLVER(EmptyQuery);
        break;
      }
      case T_FLUSH_PRIVILEGES: {
        REGISTER_STMT_RESOLVER(EmptyQuery);
        break;
      }
      case T_LOCK_TABLE: {
        REGISTER_STMT_RESOLVER(LockTable);
        break;
      }
      case T_CREATE_OUTLINE: {
        REGISTER_STMT_RESOLVER(CreateOutline);
        break;
      }
      case T_ALTER_OUTLINE: {
        REGISTER_STMT_RESOLVER(AlterOutline);
        break;
      }
      case T_DROP_OUTLINE: {
        REGISTER_STMT_RESOLVER(DropOutline);
        break;
      }
      case T_SP_CREATE: {
        REGISTER_STMT_RESOLVER(CreateProcedure);
        break;
      }
      case T_SP_ALTER: {
        REGISTER_STMT_RESOLVER(AlterProcedure);
        break;
      }
      case T_SP_CALL_STMT: {
        REGISTER_STMT_RESOLVER(CallProcedure);
        break;
      }
      case T_SF_CREATE: {
        REGISTER_STMT_RESOLVER(CreateFunction);
        break;
      }
      case T_SF_ALTER: {
        REGISTER_STMT_RESOLVER(AlterFunction);
        break;
      }
      case T_SP_DROP: {
        REGISTER_STMT_RESOLVER(DropProcedure);
        break;
      }
      case T_SF_DROP: {
        REGISTER_STMT_RESOLVER(DropFunction);
        break;
      }
      case T_SP_ANONYMOUS_BLOCK: {
        REGISTER_STMT_RESOLVER(AnonymousBlock);
        break;
      }
      case T_PACKAGE_CREATE: {
        REGISTER_STMT_RESOLVER(CreatePackage);
        break;
      }
      case T_PACKAGE_CREATE_BODY: {
        REGISTER_STMT_RESOLVER(CreatePackageBody);
        break;
      }
      case T_PACKAGE_DROP: {
        REGISTER_STMT_RESOLVER(DropPackage);
        break;
      }
      case T_MYSQL_UPDATE_HISTOGRAM:
      case T_MYSQL_DROP_HISTOGRAM:
      case T_MYSQL_ANALYZE: {
        REGISTER_STMT_RESOLVER(AnalyzeStmt);
        break;
      }
      case T_LOAD_DATA_URL:
      case T_LOAD_DATA: {
        REGISTER_STMT_RESOLVER(LoadData);
        break;
      }
      case T_SET_TABLE_COMMENT:
      case T_SET_COLUMN_COMMENT: {
        REGISTER_STMT_RESOLVER(SetComment);
        break;
      }
      case T_CREATE_ROLE: {
        REGISTER_STMT_RESOLVER(CreateRole);
        break;
      }
      case T_DROP_ROLE: {
        REGISTER_STMT_RESOLVER(DropRole);
        break;
      }
      case T_ALTER_ROLE: {
        REGISTER_STMT_RESOLVER(AlterRole);
        break;
      }
      case T_CREATE_SAVEPOINT:
      case T_ROLLBACK_SAVEPOINT:
      case T_RELEASE_SAVEPOINT: {
        REGISTER_STMT_RESOLVER(SavePoint);
        break;
      }
      case T_TG_CREATE: {
        REGISTER_STMT_RESOLVER(Trigger);
        break;
      }
      case T_TG_DROP: {
        REGISTER_STMT_RESOLVER(Trigger);
        break;
      }
      case T_TG_ALTER: {
        REGISTER_STMT_RESOLVER(Trigger);
        break;
      }
      case T_DIAGNOSTICS: {
        REGISTER_STMT_RESOLVER(GetDiagnostics);
        break;
      }
      case T_ALTER_SYSTEM_RESET_PARAMETER: {
        REGISTER_STMT_RESOLVER(ResetConfig);
        break;
      }
      case T_ALTER_SYSTEM_RESET: {
        REGISTER_STMT_RESOLVER(AlterSystemReset);
        break;
      }
      default: {
        ret = OB_NOT_SUPPORTED;
        const char *type_name = get_type_name(parse_tree.type_);
        LOG_USER_ERROR(OB_NOT_SUPPORTED, "statement type");
        break;
      }
    }  // end switch
    if (OB_SUCC(ret) && stmt->is_dml_write_stmt()) {
      // todo yanli:check leader-follower database
    }
    
    if (OB_SUCC(ret) && stmt->is_dml_stmt() && !params_.session_info_->is_varparams_sql_prepare()) {
      ObDMLStmt *dml_stmt = static_cast<ObDMLStmt*>(stmt);
      ObRawExprWrapEnumSet enum_set_wrapper(*params_.expr_factory_, params_.session_info_);
      if (OB_FAIL(enum_set_wrapper.wrap_enum_set(*dml_stmt))) {
      }
    }

    if (OB_SUCC(ret) && stmt->is_dml_stmt() && !stmt->is_explain_stmt()) {
      if (OB_FAIL(params_.query_ctx_->query_hint_.init_query_hint(params_.allocator_,
                                                                  params_.session_info_,
                                                                  static_cast<ObDMLStmt*>(stmt)))) {
      } else if (OB_FAIL(params_.query_ctx_->query_hint_.check_and_set_params_from_hint(params_,
                                                         *static_cast<ObDMLStmt*>(stmt)))) {
      }
    }

    if (OB_SUCC(ret) && stmt->is_dml_stmt() && !stmt->is_explain_stmt()) {
      bool is_contain_inner_table = false;
      bool is_contain_select_for_update = false;
      ObDMLStmt *dml_stmt = static_cast<ObDMLStmt*>(stmt);
      if (OB_FAIL(dml_stmt->check_if_contain_inner_table(is_contain_inner_table))) {
      } else if (OB_FAIL(dml_stmt->check_if_contain_select_for_update(
                  is_contain_select_for_update))) {
      } else {
        params_.query_ctx_->is_contain_inner_table_ = is_contain_inner_table;
        params_.query_ctx_->is_contain_select_for_update_ = is_contain_select_for_update;
        params_.query_ctx_->has_dml_write_stmt_ = dml_stmt->is_dml_write_stmt();
      }

    }
    if (OB_SUCC(ret)) {
      stmt::StmtType stmt_type = stmt->get_stmt_type();
      if (ObStmt::is_ddl_stmt(stmt_type, stmt->has_global_variable()) || ObStmt::is_dcl_stmt(stmt_type)) {
        ObDDLStmt *ddl_stmt = static_cast<ObDDLStmt*>(stmt);
        obcall::ObDDLArg &ddl_arg = ddl_stmt->get_ddl_arg();

        if (OB_ISNULL(params_.query_ctx_)) {
          ret = OB_ERR_UNEXPECTED;
        } else {
          ddl_arg.ddl_stmt_str_ = params_.query_ctx_->get_sql_stmt();
        }
      }
    }
  }  // end if
  return ret;
}

}  // namespace sql
}  // namespace oceanbase
