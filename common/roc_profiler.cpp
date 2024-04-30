
#include <thread>
#include <mutex>
#include <iostream>
#include <fstream>
#include "common/roc_profiler.h"

static std::string cxx_demangle(const std::string& s) {
  return s;
}

const char* GetDomainName(rocprofiler_tracer_activity_domain_t domain) {
  switch (domain) {
    case ACTIVITY_DOMAIN_ROCTX:
      return "ROCTX_DOMAIN";
      break;
    case ACTIVITY_DOMAIN_HIP_API:
      return "HIP_API_DOMAIN";
      break;
    case ACTIVITY_DOMAIN_HIP_OPS:
      return "HIP_OPS_DOMAIN";
      break;
    case ACTIVITY_DOMAIN_HSA_API:
      return "HSA_API_DOMAIN";
      break;
    case ACTIVITY_DOMAIN_HSA_OPS:
      return "HSA_OPS_DOMAIN";
      break;
    case ACTIVITY_DOMAIN_HSA_EVT:
      return "HSA_EVT_DOMAIN";
      break;
    default:
      return "";
      break;
  }
}

//static std::ofstream output_file("profile.out");
#define output_file std::cerr


[[maybe_unused]] uint32_t GetPid() {
  return 0;
  // static uint32_t pid = syscall(__NR_getpid);
  // return pid;
}

// void prepare() {
//   output_file.copyfmt(std::cout);
//   output_file.clear(std::cout.rdstate());
//   output_file.basic_ios<char>::rdbuf(std::cout.rdbuf());
// }

static std::mutex writing_lock;


// Flush function needs to be provided by the user to be used in three cases by
// the user buffer:
//   1- Application is finished
//   2- Buffer is full
//   3- Flush Interval specified by the user
void FlushTracerRecord(rocprofiler_record_tracer_t tracer_record,
                       rocprofiler_session_id_t session_id,
                       rocprofiler_buffer_id_t buffer_id = rocprofiler_buffer_id_t{0}) {
  std::lock_guard<std::mutex> lock(writing_lock);
  std::string function_name;

  if (tracer_record.domain == ACTIVITY_DOMAIN_HSA_API || ACTIVITY_DOMAIN_HIP_API) {
    const char* function_name_c = nullptr;
    CHECK_ROCPROFILER(rocprofiler_query_tracer_operation_name(
        tracer_record.domain, tracer_record.operation_id, &function_name_c));
    function_name = function_name_c ? function_name_c : "";
  }

  output_file << "Record [" << tracer_record.header.id.handle << "], Domain("
              << GetDomainName(tracer_record.domain);
  if (tracer_record.phase == ROCPROFILER_PHASE_ENTER) {
    rocprofiler_timestamp_t timestamp;
    rocprofiler_get_timestamp(&timestamp);
    output_file << "), Begin(" << timestamp.value;
  } else if (tracer_record.phase == ROCPROFILER_PHASE_EXIT) {
    rocprofiler_timestamp_t timestamp;
    rocprofiler_get_timestamp(&timestamp);
    output_file << "), End(" << timestamp.value;
  } else {
    output_file << "), Begin(" << tracer_record.timestamps.begin.value << "), End("
                << tracer_record.timestamps.end.value;
  }
  output_file << "), Correlation ID(" << tracer_record.correlation_id.value << ")";
  if (tracer_record.domain == ACTIVITY_DOMAIN_ROCTX && tracer_record.operation_id.id >= 0)
    output_file << ", ROCTX ID(" << tracer_record.operation_id.id << ")";
  if (tracer_record.domain == ACTIVITY_DOMAIN_ROCTX && tracer_record.name)
    output_file << ", ROCTX Message(" << tracer_record.name << ")";
  if (function_name.size() > 1) output_file << ", Function(" << function_name << ")";
  if (tracer_record.domain == ACTIVITY_DOMAIN_HIP_OPS && tracer_record.name)
    output_file << ", Kernel Name(" << cxx_demangle(tracer_record.name) << ")";
  output_file << std::endl;
}

void FlushProfilerRecord(const rocprofiler_record_profiler_t* profiler_record,
                         rocprofiler_session_id_t session_id, rocprofiler_buffer_id_t buffer_id) {
  std::lock_guard<std::mutex> lock(writing_lock);
  size_t name_length = 0;
  CHECK_ROCPROFILER(rocprofiler_query_kernel_info_size(ROCPROFILER_KERNEL_NAME,
                                                       profiler_record->kernel_id, &name_length));
  // Taken from rocprofiler: The size hasn't changed in  recent past
  static const uint32_t lds_block_size = 128 * 4;
  const char* kernel_name_c = "";
  if (name_length > 1) {
    kernel_name_c = static_cast<const char*>(malloc(name_length * sizeof(char)));
    CHECK_ROCPROFILER(rocprofiler_query_kernel_info(ROCPROFILER_KERNEL_NAME,
                                                    profiler_record->kernel_id, &kernel_name_c));
  }
  output_file << std::string("dispatch[") << std::to_string(profiler_record->header.id.handle)
              << "], " << std::string("gpu_id(") << std::to_string(profiler_record->gpu_id.handle)
              << "), " << std::string("queue_id(")
              << std::to_string(profiler_record->queue_id.handle) << "), "
              << std::string("queue_index(") << std::to_string(profiler_record->queue_idx.value)
              << "), " << std::string("pid(") << std::to_string(GetPid()) << "), "
              << std::string("tid(") << std::to_string(profiler_record->thread_id.value) << ")";
  output_file << ", " << std::string("grd(")
              << std::to_string(profiler_record->kernel_properties.grid_size) << "), "
              << std::string("wgr(")
              << std::to_string(profiler_record->kernel_properties.workgroup_size) << "), "
              << std::string("lds(")
              << std::to_string(
                     ((profiler_record->kernel_properties.lds_size + (lds_block_size - 1)) &
                      ~(lds_block_size - 1)))
              << "), " << std::string("scr(")
              << std::to_string(profiler_record->kernel_properties.scratch_size) << "), "
              << std::string("arch_vgpr(")
              << std::to_string(profiler_record->kernel_properties.arch_vgpr_count) << "), "
              << std::string("accum_vgpr(")
              << std::to_string(profiler_record->kernel_properties.accum_vgpr_count) << "), "
              << std::string("sgpr(")
              << std::to_string(profiler_record->kernel_properties.sgpr_count) << "), "
              << std::string("wave_size(")
              << std::to_string(profiler_record->kernel_properties.wave_size) << "), "
              << std::string("sig(")
              << std::to_string(profiler_record->kernel_properties.signal_handle);
  std::string kernel_name = cxx_demangle(kernel_name_c);
  output_file << "), " << std::string("obj(") << std::to_string(profiler_record->kernel_id.handle)
              << "), " << std::string("kernel-name(\"") << kernel_name << "\")"
              << std::string(", time(") << std::to_string(profiler_record->timestamps.begin.value)
              << ") ";

  // For Counters
  output_file << std::endl;
  if (profiler_record->counters) {
    for (uint64_t i = 0; i < profiler_record->counters_count.value; i++) {
      if (profiler_record->counters[i].counter_handler.handle > 0) {
        size_t counter_name_length = 0;
        CHECK_ROCPROFILER(rocprofiler_query_counter_info_size(
            session_id, ROCPROFILER_COUNTER_NAME, profiler_record->counters[i].counter_handler,
            &counter_name_length));
        if (counter_name_length > 1) {
          const char* name_c = static_cast<const char*>(malloc(name_length * sizeof(char)));
          CHECK_ROCPROFILER(rocprofiler_query_counter_info(
              session_id, ROCPROFILER_COUNTER_NAME, profiler_record->counters[i].counter_handler,
              &name_c));
          output_file << ", " << name_c << " ("
                      << std::to_string(profiler_record->counters[i].value.value) << ")"
                      << std::endl;
        }
      }
    }
  }
}

void FlushPCSamplingRecord(const rocprofiler_record_pc_sample_t* pc_sampling_record) {
  const auto& sample = pc_sampling_record->pc_sample;
  output_file << "dispatch[" << sample.dispatch_id.value << "], "
              << "timestamp(" << sample.timestamp.value << "), "
              << "gpu_id(" << sample.gpu_id.handle << "), "
              << "pc-sample(" << std::hex << std::showbase << sample.pc << "), "
              << "se(" << sample.se << ')' << std::endl;
}

void FlushCountersSamplerRecord(
    const rocprofiler_record_counters_sampler_t* counters_sampler_record) {
  for (uint32_t i = 0; i < counters_sampler_record->num_counters; i++) {
    output_file << ",Counter_" << i << "("
                << std::to_string(counters_sampler_record->counters[i].value.value) << ")"
                << std::endl;
  }
  output_file << std::endl;
}

int WriteBufferRecords(const rocprofiler_record_header_t* begin,
                       const rocprofiler_record_header_t* end, rocprofiler_session_id_t session_id,
                       rocprofiler_buffer_id_t buffer_id) {
  while (begin < end) {
    if (!begin) return 0;
    switch (begin->kind) {
      case ROCPROFILER_PROFILER_RECORD: {
        const rocprofiler_record_profiler_t* profiler_record =
            reinterpret_cast<const rocprofiler_record_profiler_t*>(begin);
        FlushProfilerRecord(profiler_record, session_id, buffer_id);
        break;
      }
      case ROCPROFILER_TRACER_RECORD: {
        rocprofiler_record_tracer_t* tracer_record = const_cast<rocprofiler_record_tracer_t*>(
            reinterpret_cast<const rocprofiler_record_tracer_t*>(begin));
        FlushTracerRecord(*tracer_record, session_id, buffer_id);
        break;
      }
      case ROCPROFILER_PC_SAMPLING_RECORD: {
        const rocprofiler_record_pc_sample_t* pc_sampling_record =
            reinterpret_cast<const rocprofiler_record_pc_sample_t*>(begin);
        FlushPCSamplingRecord(pc_sampling_record);
        break;
      }
      case ROCPROFILER_COUNTERS_SAMPLER_RECORD: {
        const rocprofiler_record_counters_sampler_t* counters_sampler_record =
            reinterpret_cast<const rocprofiler_record_counters_sampler_t*>(begin);
        FlushCountersSamplerRecord(counters_sampler_record);
        break;
      }
      default: {
        std::cout << "unknown record\n";
        break;
      }
    }
    rocprofiler_next_record(begin, &begin, session_id, buffer_id);
  }
  return 0;
}

RocProfilerSession::RocProfilerSession() {

  int counter_option = 0;
#if 0  
     // Initialize the tools
  CHECK_ROCPROFILER(rocprofiler_initialize());

  // Creating the session with given replay mode
  CHECK_ROCPROFILER(rocprofiler_create_session(ROCPROFILER_NONE_REPLAY_MODE, &session_id));

  // Creating Output Buffer for the data
  CHECK_ROCPROFILER(rocprofiler_create_buffer(
      session_id,
      [](const rocprofiler_record_header_t* record, const rocprofiler_record_header_t* end_record,
         rocprofiler_session_id_t session_id, rocprofiler_buffer_id_t buffer_id) {
        WriteBufferRecords(record, end_record, session_id, buffer_id);
      },
      0x999999, &buffer_id));

  // Counters Sampler Filter
  rocprofiler_filter_id_t filter_id;
  [[maybe_unused]] rocprofiler_filter_property_t property = {};

  std::vector<std::string> xgmi_counters = {"xgmi_link0_data_outbound", "xgmi_link1_data_outbound",
                                            "xgmi_link2_data_outbound", "xgmi_link3_data_outbound",
                                            "xgmi_link4_data_outbound", "xgmi_link5_data_outbound",
                                            "xgmi_link6_data_outbound", "xgmi_link7_data_outbound"};

  rocprofiler_counters_sampler_counter_input_t counters_input[2] = {
      {.name = const_cast<char*>(xgmi_counters[counter_option].c_str()),
       .type = ROCPROFILER_COUNTERS_SAMPLER_XGMI_COUNTERS}};

  uint32_t rate = 1000;
  uint32_t duration = 5000;

  rocprofiler_counters_sampler_parameters_t cs_parameters = {.counters = counters_input,
                                                             .counters_num = 1,
                                                             .sampling_rate = rate,
                                                             .sampling_duration = duration,
                                                             .gpu_agent_index = 0};
  CHECK_ROCPROFILER(rocprofiler_create_filter(
      session_id, ROCPROFILER_COUNTERS_SAMPLER,
      rocprofiler_filter_data_t{.counters_sampler_parameters = cs_parameters}, 0, &filter_id,
      property));
  CHECK_ROCPROFILER(rocprofiler_set_filter_buffer(session_id, filter_id, buffer_id));
#else
// Initialize the tools
  CHECK_ROCPROFILER(rocprofiler_initialize());

  // Creating the session with given replay mode
  CHECK_ROCPROFILER(rocprofiler_create_session(ROCPROFILER_NONE_REPLAY_MODE, &session_id));

  // Creating Output Buffer for the data
  rocprofiler_buffer_id_t buffer_id;
  CHECK_ROCPROFILER(rocprofiler_create_buffer(
      session_id,
      [](const rocprofiler_record_header_t* record, const rocprofiler_record_header_t* end_record,
         rocprofiler_session_id_t session_id, rocprofiler_buffer_id_t buffer_id) {
        WriteBufferRecords(record, end_record, session_id, buffer_id);
      },
      0x9999, &buffer_id));

  // Tracing Filter
  // std::vector<rocprofiler_tracer_activity_domain_t> apis_requested;
  // apis_requested.emplace_back(ACTIVITY_DOMAIN_HIP_API);
  // apis_requested.emplace_back(ACTIVITY_DOMAIN_HIP_OPS);
  // apis_requested.emplace_back(ACTIVITY_DOMAIN_HSA_API);
  // apis_requested.emplace_back(ACTIVITY_DOMAIN_HSA_OPS);
  // apis_requested.emplace_back(ACTIVITY_DOMAIN_ROCTX);
  // rocprofiler_filter_id_t api_tracing_filter_id;
  // CHECK_ROCPROFILER(rocprofiler_create_filter(
  //     session_id, ROCPROFILER_API_TRACE, rocprofiler_filter_data_t{&apis_requested[0]},
  //     apis_requested.size(), &api_tracing_filter_id, rocprofiler_filter_property_t{}));
  // CHECK_ROCPROFILER(rocprofiler_set_filter_buffer(session_id, api_tracing_filter_id, buffer_id));
  // CHECK_ROCPROFILER(rocprofiler_set_api_trace_sync_callback(
  //     session_id, api_tracing_filter_id,
  //     [](rocprofiler_record_tracer_t record, rocprofiler_session_id_t session_id) {
  //       FlushTracerRecord(record, session_id);
  //     }));

  // Kernel Tracing
  rocprofiler_filter_id_t kernel_tracing_filter_id;
  CHECK_ROCPROFILER(rocprofiler_create_filter(
      session_id, ROCPROFILER_DISPATCH_TIMESTAMPS_COLLECTION, rocprofiler_filter_data_t{}, 0,
      &kernel_tracing_filter_id, rocprofiler_filter_property_t{}));
  CHECK_ROCPROFILER(rocprofiler_set_filter_buffer(session_id, kernel_tracing_filter_id, buffer_id));
#endif

}

void RocProfilerSession::start() {
  CHECK_ROCPROFILER(rocprofiler_start_session(session_id));
  VLOG("Profiling session started..");
}

void RocProfilerSession::stop() {
  VLOG("Profiling session stopping..");
  CHECK_ROCPROFILER(rocprofiler_terminate_session(session_id));
  VLOG("Profiling session stopped..");
}

RocProfilerSession::~RocProfilerSession() try {
    // Deactivating session
  //CHECK_ROCPROFILER(rocprofiler_terminate_session(session_id));

  // Manual Flush user buffer request
  CHECK_ROCPROFILER(rocprofiler_flush_data(session_id, buffer_id));

  // Destroy sessions
  CHECK_ROCPROFILER(rocprofiler_destroy_session(session_id));

  // Destroy all profiling related objects(User buffer, sessions, filters,
  // etc..)
  CHECK_ROCPROFILER(rocprofiler_finalize());
}
catch(std::exception& ex) {
  VLOG("ZException: " << ex.what());
}
