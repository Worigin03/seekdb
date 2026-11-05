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
#include "parquet_reader.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <limits>
#include <sstream>

namespace oceanbase { namespace sql { namespace file_sql {
namespace {
constexpr size_t PAGE_LIMIT = 16 * 1024 * 1024;
constexpr size_t MEMORY_LIMIT = 128 * 1024 * 1024;
constexpr size_t VALUE_LIMIT = 1024 * 1024;

void parquet_require(bool valid, const char *message)
{ if (!valid) throw FileError(FileError::INVALID,std::string("file_sql: invalid Parquet: ")+message); }
void supported(bool valid, const char *message)
{ if (!valid) throw FileError(FileError::UNSUPPORTED,std::string("file_sql: unsupported Parquet: ")+message); }

struct Bytes {
  const uint8_t *data;
  size_t size, pos = 0;
  explicit Bytes(const std::string &s) : data(reinterpret_cast<const uint8_t *>(s.data())),size(s.size()) {}
  uint8_t byte() { parquet_require(pos<size,"truncated data"); return data[pos++]; }
  uint64_t little(size_t count) {
    parquet_require(count<=8 && count<=size-pos,"truncated integer");
    uint64_t value=0; for(size_t i=0;i<count;++i)value|=uint64_t(data[pos++])<<(i*8); return value;
  }
  uint64_t varint() {
    uint64_t value=0;
    for(int shift=0;shift<64;shift+=7) {
      uint8_t b=byte(); parquet_require(shift!=63 || b<=1,"varint overflow");
      value|=uint64_t(b&127)<<shift; if(!(b&128))return value;
    }
    throw FileError(FileError::INVALID,"file_sql: Parquet varint overflow");
  }
  std::string string(size_t count) {
    parquet_require(count<=PAGE_LIMIT && count<=size-pos,"truncated or excessive string");
    std::string s(reinterpret_cast<const char *>(data+pos),count);pos+=count;return s;
  }
};

struct CompactValue {
  int type=0;
  int64_t number=0;
  std::string text;
  std::vector<int> ids;
  std::vector<CompactValue> fields;
  std::vector<CompactValue> list;
  const CompactValue *find(int id) const { for(size_t i=0;i<ids.size();++i)if(ids[i]==id)return &fields[i];return nullptr; }
  const CompactValue &field(int id) const {
    const CompactValue *v=find(id);parquet_require(v!=nullptr,"missing required metadata field");return *v;
  }
  int64_t integer(int id, int64_t fallback=0) const {
    const auto *v=find(id); if(!v)return fallback;
    parquet_require(v->type>=1 && v->type<=6,"invalid integer metadata");return v->number;
  }
};

class CompactReader {
public:
  explicit CompactReader(Bytes &bytes) : b_(bytes) {}
  CompactValue value(int type,int depth=0) {
    parquet_require(depth<32 && ++nodes_<1000000,"metadata complexity limit");
    CompactValue out;out.type=type;
    if(type==1 || type==2)out.number=type==1;
    else if(type==3)out.number=static_cast<int8_t>(b_.byte());
    else if(type>=4 && type<=6) { uint64_t n=b_.varint();out.number=static_cast<int64_t>(n>>1)^-static_cast<int64_t>(n&1); }
    else if(type==7)b_.little(8);
    else if(type==8)out.text=b_.string(b_.varint());
    else if(type==9 || type==10) {
      uint8_t header=b_.byte();size_t count=header>>4;int element=header&15;
      if(count==15)count=b_.varint();
      parquet_require(count<=100000,"metadata collection limit");
      for(size_t i=0;i<count;++i)out.list.push_back(value(element,depth+1));
    } else if(type==11) {
      size_t count=b_.varint();parquet_require(count<=100000,"metadata map limit");
      if(count) { uint8_t header=b_.byte();for(size_t i=0;i<count;++i) {value(header>>4,depth+1);value(header&15,depth+1);} }
    } else if(type==12) {
      int previous=0;
      for(;;) {
        uint8_t header=b_.byte();if(header==0)break;
        int id=header>>4;
        if(id)id+=previous;
        else {uint64_t n=b_.varint();id=static_cast<int64_t>(n>>1)^-static_cast<int64_t>(n&1);}
        parquet_require(id>0,"invalid field id");previous=id;
        for(int existing:out.ids)parquet_require(existing!=id,"duplicate metadata field");
        out.ids.push_back(id);out.fields.push_back(value(header&15,depth+1));
      }
    } else parquet_require(false,"unknown compact protocol type");
    return out;
  }
private:
  Bytes &b_;size_t nodes_=0;
};

std::vector<uint32_t> hybrid(Bytes &bytes,unsigned width,size_t count)
{
  parquet_require(width<=32 && count<=VALUE_LIMIT,"invalid RLE dimensions");
  std::vector<uint32_t> result;result.reserve(count);
  while(result.size()<count) {
    uint64_t header=bytes.varint();parquet_require(header>0,"empty RLE run");
    if(!(header&1)) {
      uint64_t length=header>>1;parquet_require(length<=count-result.size()+7,"excessive RLE run");
      uint64_t v=bytes.little((width+7)/8);
      parquet_require(width==32 || v<(UINT64_C(1)<<width),"RLE value exceeds bit width");
      for(size_t i=0;i<length && result.size()<count;++i)result.push_back(v);
    } else {
      uint64_t groups=header>>1;parquet_require(groups<=((count-result.size()+7)/8),"excessive bit-packed run");
      const uint64_t total=groups*8;
      parquet_require(groups*width<=bytes.size-bytes.pos,"truncated bit-packed run");
      const size_t start=bytes.pos;
      for(uint64_t i=0;i<total;++i) {
        uint32_t v=0;
        for(unsigned bit=0;bit<width;++bit) {
          uint64_t offset=i*width+bit;
          if(bytes.data[start+offset/8]&(1U<<(offset%8)))v|=uint32_t(1)<<bit;
        }
        if(result.size()<count)result.push_back(v);
      }
      bytes.pos+=groups*width;
    }
  }
  return result;
}

uint32_t page_crc(const std::string &s)
{
  uint32_t crc=~uint32_t(0);
  for(unsigned char c:s) {crc^=c;for(int i=0;i<8;++i)crc=(crc>>1)^(0xedb88320U&-(crc&1));}
  return ~crc;
}

std::string decimal(double value)
{
  parquet_require(std::isfinite(value),"non-finite floating-point value");
  std::ostringstream out;out<<std::setprecision(17)<<value;return out.str();
}

std::string timestamp_text(int64_t micros,bool date_only)
{
  int64_t seconds=micros/1000000,fraction=micros%1000000;
  if(fraction<0) {fraction+=1000000;--seconds;}
  time_t time=static_cast<time_t>(seconds);struct tm calendar;
#if defined(_WIN32)
  parquet_require(::gmtime_s(&calendar,&time)==0,"timestamp outside calendar range");
#else
  parquet_require(::gmtime_r(&time,&calendar)!=nullptr,"timestamp outside calendar range");
#endif
  parquet_require(calendar.tm_year>=-1899 && calendar.tm_year<=8099,"timestamp outside SQL range");
  char result[96];
  if(date_only) ::snprintf(result,sizeof(result),"%04d-%02d-%02d",calendar.tm_year+1900,calendar.tm_mon+1,calendar.tm_mday);
  else ::snprintf(result,sizeof(result),"%04d-%02d-%02d %02d:%02d:%02d.%06d",calendar.tm_year+1900,calendar.tm_mon+1,calendar.tm_mday,calendar.tm_hour,calendar.tm_min,calendar.tm_sec,static_cast<int>(fraction));
  return result;
}
} // namespace

std::string ParquetReader::decode_snappy(const std::string &input,size_t expected)
{
  parquet_require(expected<=PAGE_LIMIT,"Snappy output exceeds page limit");
  Bytes bytes(input);parquet_require(bytes.varint()==expected,"Snappy length mismatch");
  std::string output;output.reserve(expected);
  while(bytes.pos<bytes.size) {
    uint8_t tag=bytes.byte();uint32_t kind=tag&3;uint64_t length=0,offset=0;
    if(kind==0) {
      length=(tag>>2)+1;
      if(length>60)length=bytes.little(length-60)+1;
      parquet_require(length<=expected-output.size(),"Snappy literal overflow");
      output+=bytes.string(length);
    } else {
      if(kind==1) {length=4+((tag>>2)&7);offset=((uint64_t(tag)&0xe0)<<3)|bytes.byte();}
      else {length=1+(tag>>2);offset=bytes.little(kind==2?2:4);}
      parquet_require(offset>0 && offset<=output.size() && length<=expected-output.size(),"invalid Snappy backreference");
      for(uint64_t i=0;i<length;++i)output.push_back(output[output.size()-offset]);
    }
  }
  parquet_require(output.size()==expected,"truncated Snappy output");return output;
}

class ParquetReader::Impl {
public:
  struct Field {int physical=0,converted=-1,length=0,unit=0;bool optional=false,unsigned_value=false;};
  struct Chunk {int codec=0;int64_t offset=0,end=0,values=0;};
  struct Group {int64_t rows=0;std::vector<Chunk> chunks;};
  struct Cursor {
    int64_t offset=0,end=0,consumed=0;
    int codec=0;
    std::vector<Cell> dictionary,page;
    size_t index=0;
  };
  Impl(FILE *file,int64_t size) : file_(file),size_(size) {footer();rescan();}
  std::string read(int64_t offset,size_t count) {
    parquet_require(offset>=0 && offset<=size_ && count<=static_cast<uint64_t>(size_-offset),"file offset out of bounds");
#if defined(_WIN32)
    parquet_require(::_fseeki64(file_,offset,SEEK_SET)==0,"cannot seek file");
#else
    parquet_require(::fseeko(file_,offset,SEEK_SET)==0,"cannot seek file");
#endif
    std::string data(count,'\0');parquet_require(::fread(&data[0],1,count,file_)==count,"short file read");return data;
  }
  void footer() {
    parquet_require(size_>=12 && read(0,4)=="PAR1","missing file magic");
    std::string tail=read(size_-8,8);parquet_require(tail.substr(4)=="PAR1","missing footer magic");
    Bytes bytes(tail);uint64_t length=bytes.little(4);
    parquet_require(length>0 && length<=PAGE_LIMIT && length<=static_cast<uint64_t>(size_-12),"footer length out of bounds");
    footer_offset_=size_-8-length;
    std::string encoded=read(footer_offset_,length);Bytes input(encoded);CompactReader compact(input);CompactValue meta=compact.value(12);
    parquet_require(input.pos==input.size,"trailing footer bytes");
    const auto &schema=meta.field(2).list;
    parquet_require(!schema.empty() && schema.size()<=4097 && schema[0].integer(5)==static_cast<int64_t>(schema.size()-1),"invalid flat schema");
    for(size_t i=1;i<schema.size();++i) {
      const auto &s=schema[i];Field field;Column column;
      supported(s.find(1)!=nullptr && s.integer(5)==0 && s.integer(3)<2,"nested or repeated fields");
      field.physical=s.integer(1);field.length=s.integer(2);field.optional=s.integer(3)==1;field.converted=s.integer(6,-1);
      column.name=s.field(4).text;column.nullable=field.optional;
      parquet_require(!column.name.empty() && column.name.size()<=256,"invalid column name");
      for(const auto &old:columns_)parquet_require(old.name!=column.name,"duplicate column name");
      const auto *logical=s.find(10);
      if(logical && logical->find(6))field.converted=6;
      if(logical && logical->find(8)) {
        const auto &units=logical->field(8).field(2);
        field.unit=units.find(1)?1:units.find(2)?2:units.find(3)?3:0;
        parquet_require(field.unit!=0,"invalid timestamp unit");
      }
      if(field.converted==9)field.unit=1;
      if(field.converted==10)field.unit=2;
      field.unsigned_value=field.converted>=11 && field.converted<=14;
      if(logical && logical->find(10))field.unsigned_value=!logical->field(10).integer(2,1);
      supported(field.physical>=0 && field.physical<=7,"physical type");
      supported(field.converted!=5 && !(logical && logical->find(5)),"decimal fields");
      if(field.converted==6)column.type=Type::DATE;
      else if(field.unit || field.physical==3) {
        column.type=Type::DATETIME;
        column.datetime_scale=field.unit==1?3:6;
      }
      else if(field.physical==0)column.type=Type::BOOLEAN;
      else if(field.physical==1 || field.physical==2)column.type=field.unsigned_value && field.physical==2?Type::VARCHAR:Type::BIGINT;
      else if(field.physical==4 || field.physical==5)column.type=Type::DOUBLE;
      else column.type=Type::VARCHAR;
      if(field.physical==7)parquet_require(field.length>0 && field.length<=static_cast<int>(PAGE_LIMIT),"invalid fixed-width string");
      fields_.push_back(field);columns_.push_back(column);
    }
    parquet_require(!columns_.empty(),"empty schema");
    int64_t rows=0;
    for(const auto &rg:meta.field(4).list) {
      Group group;group.rows=rg.integer(3,-1);parquet_require(group.rows>=0,"invalid row group count");
      const auto &chunks=rg.field(1).list;parquet_require(chunks.size()==columns_.size(),"column chunk count mismatch");
      for(size_t i=0;i<chunks.size();++i) {
        const auto &chunk=chunks[i];supported(!chunk.find(1) || chunk.field(1).text.empty(),"external column chunks");
        supported(!chunk.find(8) && !chunk.find(9),"encrypted column chunks");
        const auto &m=chunk.field(3);Chunk c;c.codec=m.integer(4);c.values=m.integer(5,-1);c.offset=m.integer(9,-1);
        parquet_require(m.integer(1)==fields_[i].physical && c.values==group.rows,"column metadata mismatch");
        const auto &path=m.field(3).list;parquet_require(path.size()==1 && path[0].text==columns_[i].name,"column path mismatch");
        if(m.find(11))c.offset=std::min(c.offset,m.integer(11));
        int64_t size=m.integer(7,-1);
        parquet_require(c.offset>=4 && size>=0 && c.offset<=footer_offset_ && size<=footer_offset_-c.offset,"column chunk outside file");
        c.end=c.offset+size;group.chunks.push_back(c);
      }
      parquet_require(group.rows<=std::numeric_limits<int64_t>::max()-rows,"row count overflow");rows+=group.rows;groups_.push_back(std::move(group));
    }
    parquet_require(rows==meta.integer(3,-1),"file row count mismatch");
    selected_.assign(columns_.size(),true);
  }
  Cell plain(Bytes &data,const Field &field,size_t index) {
    Cell cell;cell.is_null=false;uint64_t raw=0;int64_t integer=0;
    switch(field.physical) {
      case 0: {
        parquet_require(index/8<data.size,"truncated boolean values");cell.text=(data.data[index/8]&(1<<(index%8)))?"true":"false";return cell;
      }
      case 1:raw=data.little(4);integer=field.unsigned_value?static_cast<int64_t>(raw):static_cast<int32_t>(raw);break;
      case 2:raw=data.little(8);integer=static_cast<int64_t>(raw);break;
      case 3: {
        uint64_t nanos=data.little(8);int64_t days=static_cast<int64_t>(data.little(4))-2440588;
        parquet_require(nanos<UINT64_C(86400000000000) && days>=-719162 && days<=2932896,"invalid INT96 timestamp");
        cell.text=timestamp_text(days*INT64_C(86400000000)+nanos/1000,false);return cell;
      }
      case 4: {
        uint32_t bits=data.little(4);float value;std::memcpy(&value,&bits,4);cell.text=decimal(value);return cell;
      }
      case 5: {
        uint64_t bits=data.little(8);double value;std::memcpy(&value,&bits,8);cell.text=decimal(value);return cell;
      }
      case 6:cell.text=data.string(data.little(4));return cell;
      case 7:cell.text=data.string(field.length);return cell;
    }
    if(field.converted==6) {
      parquet_require(integer>=-719162 && integer<=2932896,"date outside SQL range");cell.text=timestamp_text(integer*INT64_C(86400000000),true);
    } else if(field.unit) {
      int64_t micros=integer;
      if(field.unit==1) {parquet_require(integer>=INT64_MIN/1000 && integer<=INT64_MAX/1000,"timestamp overflow");micros=integer*1000;}
      else if(field.unit==3) {micros=integer/1000;if(integer%1000<0)--micros;}
      cell.text=timestamp_text(micros,false);
    } else cell.text=field.unsigned_value && field.physical==2?std::to_string(raw):std::to_string(integer);
    return cell;
  }
  std::string decompress(int codec,const std::string &compressed,size_t size) {
    supported(codec==0 || codec==1,"compression codec (only uncompressed and Snappy are supported)");
    if(codec==1)return ParquetReader::decode_snappy(compressed,size);
    parquet_require(compressed.size()==size,"uncompressed page size mismatch");return compressed;
  }
  void load_page(size_t column,Cursor &cursor) {
    const Field &field=fields_[column];
    while(cursor.index>=cursor.page.size()) {
      parquet_require(cursor.offset<cursor.end,"not enough data pages");
      const size_t header_limit=std::min<int64_t>(64*1024,cursor.end-cursor.offset);
      std::string encoded_header=read(cursor.offset,header_limit);Bytes bytes(encoded_header);CompactReader compact(bytes);CompactValue header=compact.value(12);
      cursor.offset+=bytes.pos;
      const int type=header.integer(1,-1);const int64_t plain_size=header.integer(2,-1),compressed_size=header.integer(3,-1);
      parquet_require(plain_size>=0 && plain_size<=PAGE_LIMIT && compressed_size>=0 && compressed_size<=PAGE_LIMIT
                      && compressed_size<=cursor.end-cursor.offset,"invalid data page size");
      std::string compressed=read(cursor.offset,compressed_size);cursor.offset+=compressed_size;
      if(header.find(4))parquet_require(page_crc(compressed)==static_cast<uint32_t>(header.integer(4)),"page checksum mismatch");
      if(type==1)continue;
      if(type==2) {
        const auto &dict=header.field(7);int64_t count=dict.integer(1,-1);
        parquet_require(count>=0 && count<=VALUE_LIMIT && cursor.dictionary.empty(),"invalid dictionary page");
        supported(dict.integer(2)==0 || dict.integer(2)==2,"dictionary page encoding");
        std::string decoded=decompress(cursor.codec,compressed,plain_size);Bytes data(decoded);
        for(int64_t i=0;i<count;++i)cursor.dictionary.push_back(plain(data,field,i));
        memory_check();continue;
      }
      supported(type==0 || type==3,"data page type");
      const auto &page=header.field(type==0?5:8);
      int64_t count=page.integer(1,-1);int encoding=page.integer(type==0?2:4,-1);
      parquet_require(count>0 && count<=VALUE_LIMIT,"invalid data page value count");
      std::string decoded;std::vector<uint32_t> definitions;
      if(type==0) {
        decoded=decompress(cursor.codec,compressed,plain_size);Bytes data(decoded);
        if(field.optional) {
          supported(page.integer(3)==3,"definition level encoding");
          size_t length=data.little(4);std::string levels=data.string(length);Bytes level_bytes(levels);definitions=hybrid(level_bytes,1,count);
        }
        decoded.erase(0,data.pos);
      } else {
        int64_t defs=page.integer(5,-1),reps=page.integer(6,-1);
        parquet_require(defs>=0 && reps==0 && defs<=compressed_size && defs<=plain_size,"invalid V2 level lengths");
        if(field.optional) {std::string levels=compressed.substr(0,defs);Bytes data(levels);definitions=hybrid(data,1,count);}
        std::string payload=compressed.substr(defs);
        decoded=page.integer(7,1)?decompress(cursor.codec,payload,plain_size-defs):decompress(0,payload,plain_size-defs);
      }
      if(!field.optional)definitions.assign(count,1);
      const size_t nonnull=std::count(definitions.begin(),definitions.end(),1);
      if(type==3)parquet_require(count-page.integer(2,-1)==static_cast<int64_t>(nonnull),"null count mismatch");
      Bytes data(decoded);std::vector<uint32_t> indices;
      if(encoding==2 || encoding==8) {
        parquet_require(!cursor.dictionary.empty() || nonnull==0,"missing dictionary");
        if(nonnull) {unsigned width=data.byte();indices=hybrid(data,width,nonnull);}
      } else supported(encoding==0,"value encoding (only plain and dictionary are supported)");
      cursor.page.clear();cursor.page.reserve(count);cursor.index=0;size_t nonnull_index=0;
      size_t retained=memory_usage();
      parquet_require(retained<=MEMORY_LIMIT,"scan memory limit exceeded");
      for(size_t i=0;i<static_cast<size_t>(count);++i) {
        parquet_require(definitions[i]<=1,"invalid definition level");
        if(definitions[i]==0)cursor.page.push_back(Cell());
        else if(encoding==2 || encoding==8) {
          uint32_t index=indices[nonnull_index++];parquet_require(index<cursor.dictionary.size(),"dictionary index out of bounds");
          const Cell &value=cursor.dictionary[index];
          parquet_require(value.text.size()<=MEMORY_LIMIT-retained,"dictionary expansion exceeds scan memory limit");
          retained+=value.text.size();cursor.page.push_back(value);
        } else cursor.page.push_back(plain(data,field,nonnull_index++));
      }
      memory_check();
    }
  }
  size_t memory_usage() const {
    size_t size=0;
    for(const auto &cursor:cursors_) {
      for(const auto *cells:{&cursor.page,&cursor.dictionary}) {
        size+=cells->capacity()*sizeof(Cell);
        for(const auto &cell:*cells) size+=cell.text.capacity();
      }
    }
    return size;
  }
  void memory_check() const {parquet_require(memory_usage()<=MEMORY_LIMIT,"scan memory limit exceeded");}
  void rescan() {group_=0;row_=0;cursors_.clear();}
  bool next(std::vector<Cell> &row) {
    while(group_<groups_.size() && row_>=groups_[group_].rows) {++group_;row_=0;cursors_.clear();}
    if(group_>=groups_.size())return false;
    if(cursors_.empty()) {
      for(const Chunk &chunk:groups_[group_].chunks) {Cursor cursor;cursor.offset=chunk.offset;cursor.end=chunk.end;cursor.codec=chunk.codec;cursors_.push_back(std::move(cursor));}
    }
    row.assign(columns_.size(),Cell());
    for(size_t i=0;i<columns_.size();++i) {
      if(!selected_[i])continue;
      Cursor &cursor=cursors_[i];load_page(i,cursor);row[i]=cursor.page[cursor.index++];++cursor.consumed;
      parquet_require(cursor.consumed<=groups_[group_].rows,"too many column values");
    }
    ++row_;
    if(row_==groups_[group_].rows) {
      for(size_t i=0;i<cursors_.size();++i) if(selected_[i]) {
        parquet_require(cursors_[i].consumed==row_ && cursors_[i].index==cursors_[i].page.size()
                        && cursors_[i].offset==cursors_[i].end,"column chunk contains surplus values or pages");
      }
    }
    return true;
  }
  FILE *file_;int64_t size_,footer_offset_=0,row_=0;size_t group_=0;
  std::vector<Column> columns_;std::vector<Field> fields_;std::vector<Group> groups_;
  std::vector<bool> selected_;std::vector<Cursor> cursors_;
};

ParquetReader::ParquetReader(FILE *file,int64_t size) : impl_(new Impl(file,size)) {}
ParquetReader::~ParquetReader() = default;
const std::vector<Column> &ParquetReader::columns() const {return impl_->columns_;}
void ParquetReader::project(const std::vector<int64_t> &columns) {
  impl_->selected_.assign(impl_->columns_.size(),false);
  for(int64_t column:columns) {parquet_require(column>=0 && column<static_cast<int64_t>(impl_->columns_.size()),"projection outside schema");impl_->selected_[column]=true;}
}
void ParquetReader::rescan() {impl_->rescan();}
bool ParquetReader::next(std::vector<Cell> &row) {return impl_->next(row);}
}}}
