#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>


extern "C" {
  void startCapture() {
    if (![[NSProcessInfo processInfo].environment[@"METAL_CAPTURE_ENABLED"] boolValue]) {
      NSLog(@"METAL_CAPTURE_ENABLED is not set. Please set it to 1 to enable Metal capture.");
      return;
    }
    
    MTLCaptureDescriptor *descriptor = [[MTLCaptureDescriptor alloc] init];
    descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
    descriptor.outputURL = [NSURL fileURLWithPath:@"gpu.cpp.gputrace"];

    NSFileManager *fileManager = [NSFileManager defaultManager];
    if ([fileManager fileExistsAtPath:@"gpu.cpp.gputrace"]) {
      NSError *error = nil;
      [fileManager removeItemAtPath:@"gpu.cpp.gputrace" error:&error];
      if (error) {
        NSLog(@"Error deleting existing gpu.cpp.gputrace directory: %@", error);
        return;
      } else {
        NSLog(@"Deleted existing gpu.cpp.gputrace directory.");
      }
    }

    NSError *error = nil;
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      NSLog(@"MTLCreateSystemDefaultDevice returned nil. Metal may not be supported on this system.");
      return;
    }
    descriptor.captureObject = device;
    
    BOOL success = [MTLCaptureManager.sharedCaptureManager startCaptureWithDescriptor:descriptor error:&error];
    if (!success) {
        NSLog(@" error capturing mtl => %@ ", [error localizedDescription] );
    }
  }

  void stopCapture() {
    [MTLCaptureManager.sharedCaptureManager stopCapture];
  }
}
