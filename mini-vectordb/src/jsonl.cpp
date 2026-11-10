// JSON Lines transport for the competition driver.
#include "engine.h"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <string_view>
#include <type_traits>

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

namespace {

using mvdb::Database;
using Writer = rapidjson::Writer<rapidjson::StringBuffer>;

void writeString(Writer& writer, const std::string& value) {
  writer.String(value.data(), value.size());
}

void writeCell(Writer& writer, const Database::ResultCell& cell) {
  std::visit([&](const auto& value) {
    using T = std::decay_t<decltype(value)>;
    if constexpr (std::is_same_v<T, int64_t>) {
      writer.Int64(value);
    } else if constexpr (std::is_same_v<T, std::string>) {
      writeString(writer, value);
    } else if constexpr (std::is_same_v<T, std::vector<float>>) {
      writer.StartArray();
      for (float component : value) writer.Double(component);
      writer.EndArray();
    } else {
      if (std::isfinite(value)) writer.Double(value);
      else writer.Null();
    }
  }, cell);
}

void writeResult(Writer& writer, const Database::ResultSet& result) {
  if (!result.hasResult) {
    writer.Key("affected_rows");
    writer.Int64(result.affected);
    return;
  }
  writer.Key("columns");
  writer.StartArray();
  for (const auto& header : result.headers) writeString(writer, header);
  writer.EndArray();
  writer.Key("rows");
  writer.StartArray();
  for (const auto& row : result.rows) {
    writer.StartArray();
    for (const auto& cell : row) writeCell(writer, cell);
    writer.EndArray();
  }
  writer.EndArray();
}

std::string errorCode(const mvdb::SqlError& error) {
  switch (error.code) {
    case 1062: return "duplicate_key";
    case 1235: return "unsupported_sql";
    case 1366:
      if (std::string_view(error.what()).find("dimension mismatch") != std::string_view::npos)
        return "dimension_mismatch";
      return "invalid_value";
    default: return std::to_string(error.code);
  }
}

void send(const rapidjson::StringBuffer& response) {
  fwrite(response.GetString(), 1, response.GetSize(), stdout);
  fputc('\n', stdout);
  fflush(stdout);
}

}  // namespace

int runJsonl(Database& db) {
  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);
  fputs("{\"type\":\"ready\"}\n", stdout);
  fflush(stdout);

  std::string line;
  while (std::getline(std::cin, line)) {
    rapidjson::Document request;
    request.ParseInsitu<rapidjson::kParseValidateEncodingFlag |
                        rapidjson::kParseFullPrecisionFlag>(line.data());

    const rapidjson::Value* id = nullptr;
    Database::ResultSet result;
    std::string code, message;
    bool shutdown = false;
    try {
      if (request.HasParseError() || !request.IsObject())
        throw std::invalid_argument("expected a JSON request object");
      const auto requestId = request.FindMember("id");
      if (requestId == request.MemberEnd())
        throw std::invalid_argument("missing request id");
      id = &requestId->value;
      const auto op = request.FindMember("op");
      if (op == request.MemberEnd() || !op->value.IsString())
        throw std::invalid_argument("missing operation");
      const std::string_view operation(op->value.GetString(), op->value.GetStringLength());
      if (operation == "shutdown") {
        db.close();
        shutdown = true;
      } else if (operation == "execute") {
        const auto sql = request.FindMember("sql");
        const auto mode = request.FindMember("search_mode");
        if (sql == request.MemberEnd() || !sql->value.IsString())
          throw std::invalid_argument("missing SQL string");
        if (mode == request.MemberEnd() || !mode->value.IsString())
          throw std::invalid_argument("missing search mode");
        const std::string searchMode(mode->value.GetString(), mode->value.GetStringLength());
        if (searchMode != "exact" && searchMode != "ann")
          throw std::invalid_argument("unknown search mode");
        db.execute(std::string(sql->value.GetString(), sql->value.GetStringLength()), result, searchMode);
      } else {
        throw std::invalid_argument("unknown operation");
      }
    } catch (const mvdb::SqlError& error) {
      code = errorCode(error);
      message = error.what();
    } catch (const std::invalid_argument& error) {
      code = "invalid_request";
      message = error.what();
    } catch (const std::exception& error) {
      code = "internal_error";
      message = error.what();
    }

    rapidjson::StringBuffer response;
    Writer writer(response);
    writer.StartObject();
    writer.Key("id");
    if (id) id->Accept(writer);
    else writer.Null();
    writer.Key("ok");
    writer.Bool(code.empty());
    if (!code.empty()) {
      writer.Key("error");
      writer.StartObject();
      writer.Key("code");
      writeString(writer, code);
      writer.Key("message");
      writeString(writer, message);
      writer.EndObject();
    } else if (!shutdown) {
      writeResult(writer, result);
    }
    writer.EndObject();
    send(response);
    if (shutdown) return 0;
  }
  return std::cin.bad() ? 1 : 0;
}
