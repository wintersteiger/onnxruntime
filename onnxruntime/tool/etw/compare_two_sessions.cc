#include <iostream>
#include "eparser.h"
#include "TraceSession.h"

#ifdef _WIN32
#include <tchar.h>
#else
#define TCHAR char
#define _tmain main
#endif

#ifdef _WIN32
#include "getopt.h"
#else
#include <getopt.h>
#include <thread>
#endif

static const GUID OrtProviderGuid = {0x54d81939, 0x62a0, 0x4dc0, {0xbf, 0x32, 0x3, 0x5e, 0xbd, 0xc7, 0xbc, 0xe9}};

int fetch_data(TCHAR* filename, ProfilingInfo& context) {
  TraceSession session;
  session.AddHandler(OrtProviderGuid, OrtEventHandler, &context);
  session.InitializeEtlFile(filename, nullptr);
  ULONG status = ProcessTrace(&session.traceHandle_, 1, 0, 0);
  if (status != ERROR_SUCCESS && status != ERROR_CANCELLED) {
    std::cout << "OpenTrace failed with " << status << std::endl;
    session.Finalize();
    return -1;
  }
  session.Finalize();
  return 0;
}

template <typename T>
std::pair<double, double> CalcMeanAndStdSquare(const std::vector<T>& input) {
  T sum = 0;
  T sum_square = 0;
  const size_t N = input.size();
  const size_t len = input.size();
  for (T t : input) {
    sum += t;
    sum_square += t * t;
  }
  double mean = ((double)sum) / input.size();
  double std = (sum_square - N * mean * mean) / (N - 1);
  return std::make_pair(mean, std);
}

template <typename T>
double CalcTValue(const std::vector<T>& input1, const std::vector<T>& input2){
  auto p1 = CalcMeanAndStdSquare(input1);
  auto p2 = CalcMeanAndStdSquare(input2);
  auto diff_mean = p1.first - p2.first;
  size_t n1 = input1.size();
  size_t n2 = input2.size();
  auto sdiff = ((n1 - 1) * p1.second + (n2 - 1) * p2.second)/(n1+n2-2);
  sdiff *= ((double)1)/n1 + ((double)1)/n2;
  double result = diff_mean / std::sqrt(sdiff);
  return result;
}

int real_main(int argc, TCHAR* argv[]) {
  if (argc < 3) {
    printf("error\n");
    return -1;
  }
 
  ProfilingInfo context1;
  int ret = fetch_data(argv[1], context1);
  if (ret != 0) return ret;
  ProfilingInfo context2;
  ret = fetch_data(argv[2], context2);
  if (ret != 0) return ret;
  double t = CalcTValue(context1.time_per_run, context2.time_per_run);
  return 0;
}

int _tmain(int argc, TCHAR* argv[]) {
  int retval = -1;
  try {
    retval = real_main(argc, argv);
  } catch (std::exception& ex) {
    fprintf(stderr, "%s\n", ex.what());
    retval = -1;
  }
  return retval;
}