#include <cstdio>
#include <cstdlib>
#include <curl/curl.h>
#include <filesystem>
#include <map>
#include <set>
#include <string>

std::string getOSName() {
#ifdef _WIN32
  return "Windows 32-bit";
#elif _WIN64
  return "Windows 64-bit";
#elif __APPLE__ || __MACH__
  return "macOS";
#elif __linux__
  return "Linux";
#elif __FreeBSD__
  return "FreeBSD";
#elif __unix || __unix__
  return "Unix";
#else
  return "Other";
#endif
}

static size_t totalDownloaded = 0;

size_t writeCallback(void *contents, size_t size, size_t nmemb, void *userp) {
  ((std::string *)userp)->append((char *)contents, size * nmemb);
  totalDownloaded += size * nmemb;
  printf("\rDownloaded %lu MB", totalDownloaded / 1024 / 1024);
  return size * nmemb;
}

bool downloadFile(const std::string &url, const std::string &outputFilename) {
  CURL *curl = curl_easy_init();
  if (!curl)
    return false;
  std::string buffer;
  long httpCode = 0;
  char errbuf[CURL_ERROR_SIZE];
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buffer);
  curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, errbuf);
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER,
                   0L); // Only for testing! Remove in production.
  CURLcode res = curl_easy_perform(curl);
  if (res != CURLE_OK) {
    fprintf(stderr, "curl_easy_perform() failed: %s\n",
            curl_easy_strerror(res));
    fprintf(stderr, "Error buffer: %s\n", errbuf);
    curl_easy_cleanup(curl);
    return false;
  }
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &httpCode);
  curl_easy_cleanup(curl);
  if (httpCode == 200) {
    FILE *outFile = fopen(outputFilename.c_str(), "wb");
    if (!outFile) {
      fprintf(stderr, "Failed to open file for writing: %s\n",
              outputFilename.c_str());
      return false;
    }
    size_t written = fwrite(buffer.c_str(), 1, buffer.size(), outFile);
    fclose(outFile);
    if (written != buffer.size()) {
      fprintf(stderr, "Failed to write entire buffer to file\n");
      return false;
    }
    return true;
  } else {
    fprintf(stderr, "HTTP error code: %ld\n", httpCode);
    return false;
  }
}

void checkOS(const std::string &osName) {
  printf("\nChecking System\n");
  printf("===============\n\n");
  printf("  Operating System : %s\n", osName.c_str());
  // initialize supported to const set of one element
  std::set<std::string> supported;
  supported.insert("macOS");
  if (supported.find(getOSName()) == supported.end()) {
    printf("Unsupported operating system\n");
    exit(1);
  }
}

void downloadDawn(const std::string& osName)
{
  printf("\nDownload Dawn Library\n");
  printf("=====================\n\n");

  std::map<std::string, std::string> outfileMap = {
      {"macOS", "third_party/lib/libdawn.dylib"}};
  std::map<std::string, std::string> urlMap = {
      {"macOS", "https://github.com/austinvhuang/dawn-artifacts/releases/"
                "download/prerelease/libdawn.dylib"}};
  std::string outfile = outfileMap.at(getOSName());
  std::string url = urlMap.at(getOSName());

  printf("  URL              : %s\n", url.c_str());
  printf("  Download File    : %s\n\n", outfile.c_str());
  printf("  Downloading ...\n\n");

  // check if destination file exists using c lib
  FILE *file = fopen(outfile.c_str(), "r");
  if (file) {
    fclose(file);
    printf("  File %s already exists, skipping.\n", outfile.c_str());
    return;
  }

  if (downloadFile(url, outfile)) {
    printf("\n  Downloaded %s\n", outfile.c_str());
  } else {
    printf("\n  Failed to download %s\n", outfile.c_str());
  }
}

void setenv(const std::string& osName) {
  printf("\nEnvironment Setup\n");
  printf("=================\n\n");
  // get current directory
  std::string currentDir = std::filesystem::current_path();
  std::string libDir = currentDir + "/third_party/lib";
  if (osName == "macOS") {
    // print what to add to path
    printf("  Before running the program, run the following command or add it to your shell profile:\n");
    printf("  export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:%s\n", libDir.c_str());
    // write export to source script
    FILE *file = fopen("source", "w");
    if (!file) {
      fprintf(stderr, "  Failed to open file for writing: %s\n", "source.sh");
      return;
    }
    fprintf(file, "export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:%s\n", libDir.c_str());
    fclose(file);
  }
}

int main() {
  std::string osName = getOSName();
  checkOS(osName);
  downloadDawn(osName);
  setenv(osName);
  printf("\n");
  return 0;
}
