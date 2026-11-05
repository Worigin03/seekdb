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
#include <string>
#include "sql/engine/expr/ob_expr_ai/ob_ai_func_utils.h"

using namespace oceanbase;
using namespace oceanbase::common;

class AIFunctions : public ::testing::Test
{
protected:
  ObJsonObject *object(const char *text)
  {
    ObJsonObject *result = nullptr;
    ObString input(text);
    EXPECT_EQ(OB_SUCCESS, ObAIFuncJsonUtils::get_json_object_form_str(allocator_, input, result));
    return result;
  }
  ObArenaAllocator allocator_;
};

TEST_F(AIFunctions, repeated_placeholders_expand_beyond_argument_size)
{
  const std::string value(128 * 1024, 'x');
  ObJsonObject *prompt = nullptr;
  ObJsonArray *args = nullptr;
  ObJsonString *arg = nullptr;
  ObString text("{0}{0}{0}{0}");
  ObString input(value.size(), value.data());
  ObString output;
  ASSERT_EQ(OB_SUCCESS, ObAIFuncJsonUtils::get_json_array(allocator_, args));
  ASSERT_EQ(OB_SUCCESS, ObAIFuncJsonUtils::get_json_string(allocator_, input, arg));
  ASSERT_EQ(OB_SUCCESS, args->append(arg));
  ASSERT_EQ(OB_SUCCESS, ObAIFuncPromptObjectUtils::construct_prompt_object(allocator_, text, args, prompt));
  ASSERT_EQ(OB_SUCCESS, ObAIFuncPromptObjectUtils::replace_all_str_args_in_template(allocator_, prompt, output));
  EXPECT_EQ(value.size() * 4, output.length());
  EXPECT_EQ(value + value + value + value, std::string(output.ptr(), output.length()));
}

TEST_F(AIFunctions, nested_prompt_and_json_values)
{
  ObJsonObject *prompt = object(R"({"template":"{0}/{1}/{2}","args":[[1,true],{"k":2},{"template":"hello {0}","args":["world"]}]})");
  ASSERT_NE(nullptr, prompt);
  EXPECT_TRUE(ObAIFuncPromptObjectUtils::is_valid_prompt_object(prompt));
  ObString output;
  ASSERT_EQ(OB_SUCCESS, ObAIFuncPromptObjectUtils::replace_all_str_args_in_template(allocator_, prompt, output));
  const std::string rendered(output.ptr(), output.length());
  EXPECT_NE(std::string::npos, rendered.find("hello world"));
  EXPECT_NE(std::string::npos, rendered.find("true"));
  EXPECT_NE(std::string::npos, rendered.find("\"k\""));
}

TEST_F(AIFunctions, invalid_placeholder_and_invalid_prompt)
{
  ObString output;
  EXPECT_EQ(OB_INVALID_ARGUMENT, ObAIFuncPromptObjectUtils::replace_all_str_args_in_template(
      allocator_, object(R"({"template":"{999999999999999999999999999999}","args":["x"]})"), output));
  EXPECT_EQ(OB_INVALID_ARGUMENT, ObAIFuncPromptObjectUtils::replace_all_str_args_in_template(
      allocator_, object(R"({"template":"{1}","args":["x"]})"), output));
  EXPECT_EQ(OB_INVALID_ARGUMENT, ObAIFuncPromptObjectUtils::replace_all_str_args_in_template(
      allocator_, object(R"({"template":1,"args":[]})"), output));
}

TEST_F(AIFunctions, embeddings_restore_input_order)
{
  ObOpenAIUtils::ObOpenAIEmbed provider;
  ObIJsonBase *result = nullptr;
  ASSERT_EQ(OB_SUCCESS, provider.parse_output(allocator_, object(
      R"({"data":[{"index":1,"embedding":[3,4]},{"index":0,"embedding":[1,2]}]})"), result));
  auto *array = static_cast<ObJsonArray *>(result);
  ASSERT_EQ(2, array->element_count());
  EXPECT_EQ(1, static_cast<ObJsonArray *>(array->get_value(0))->get_value(0)->get_int());
  EXPECT_EQ(3, static_cast<ObJsonArray *>(array->get_value(1))->get_value(0)->get_int());
}

TEST_F(AIFunctions, malformed_embedding_responses)
{
  ObOpenAIUtils::ObOpenAIEmbed provider;
  ObIJsonBase *result = nullptr;
  const char *invalid[] = {
    R"({"data":{}})", R"({"data":[]})",
    R"({"data":[{"index":0,"embedding":[1]},{"index":0,"embedding":[2]}]})",
    R"({"data":[{"index":-1,"embedding":[1]}]})",
    R"({"data":[{"index":1,"embedding":[1]}]})",
    R"({"data":[{"index":"0","embedding":[1]}]})",
    R"({"data":[{"index":0,"embedding":{}}]})"
  };
  for (const char *text : invalid) {
    EXPECT_NE(OB_SUCCESS, provider.parse_output(allocator_, object(text), result)) << text;
  }
}

TEST_F(AIFunctions, completion_requires_nonempty_text)
{
  ObOpenAIUtils::ObOpenAIComplete provider;
  ObIJsonBase *result = nullptr;
  EXPECT_EQ(OB_SUCCESS, provider.parse_output(allocator_, object(
      R"({"choices":[{"message":{"content":"ok"}}]})"), result));
  const char *invalid[] = {
    R"({"choices":[]})", R"({"choices":[{"message":{"content":null}}]})",
    R"({"choices":[{"message":{"content":[]}}]})",
    R"({"choices":[{"message":{"content":""}}]})"
  };
  for (const char *text : invalid) {
    EXPECT_NE(OB_SUCCESS, provider.parse_output(allocator_, object(text), result)) << text;
  }
}

TEST_F(AIFunctions, ollama_is_native_and_non_streaming)
{
  ObOllamaUtils::ObOllamaComplete provider;
  ObString model("model"), system("system"), input("hello");
  ObJsonObject *body = nullptr;
  ASSERT_EQ(OB_SUCCESS, provider.get_body(allocator_, model, system, input, nullptr, body));
  ASSERT_NE(nullptr, body->get_value("stream"));
  EXPECT_EQ(ObJsonNodeType::J_BOOLEAN, body->get_value("stream")->json_type());
  EXPECT_FALSE(body->get_value("stream")->get_boolean());
  EXPECT_EQ(input, static_cast<ObJsonString *>(body->get_value("prompt"))->get_str());
  EXPECT_EQ(system, static_cast<ObJsonString *>(body->get_value("system"))->get_str());
  ObString key;
  ObArray<ObString> headers;
  EXPECT_EQ(OB_SUCCESS, provider.get_header(allocator_, key, headers));
}

TEST_F(AIFunctions, rerank_validates_and_sorts_ties_by_original_index)
{
  ObJsonObject *response = object(R"({"results":[{"index":2,"relevance_score":1},{"index":1,"relevance_score":2},{"index":0,"relevance_score":2}]})");
  ObJsonArray *result = nullptr;
  ASSERT_EQ(OB_SUCCESS, ObAIFuncUtils::validate_rerank_output(allocator_, response->get_value("results"), 3, result));
  ASSERT_EQ(3, result->element_count());
  for (int64_t i = 0; i < 3; ++i) {
    EXPECT_EQ(i, static_cast<ObJsonObject *>(result->get_value(i))->get_value("index")->get_int());
  }
  response = object(R"({"results":[{"index":0,"relevance_score":1},{"index":0,"relevance_score":2}]})");
  EXPECT_EQ(OB_INVALID_DATA, ObAIFuncUtils::validate_rerank_output(allocator_, response->get_value("results"), 2, result));
  response = object(R"({"results":[{"index":0,"relevance_score":"1"}]})");
  EXPECT_EQ(OB_INVALID_DATA, ObAIFuncUtils::validate_rerank_output(allocator_, response->get_value("results"), 1, result));
}

TEST_F(AIFunctions, ollama_empty_key_needs_no_decryption)
{
  share::ObAiModelEndpointInfo endpoint;
  ObString name("local");
  ObJsonObject *config=object(R"({"ai_model_name":"local","url":"http://127.0.0.1/api/generate","provider":"OLLAMA","access_key":""})");
  ASSERT_NE(nullptr,config);
  ASSERT_EQ(OB_SUCCESS,endpoint.parse_from_json_base(allocator_,name,*config));
  ObString key("stale");
  ASSERT_EQ(OB_SUCCESS,endpoint.get_unencrypted_access_key(allocator_,key));
  EXPECT_TRUE(key.empty());
}
