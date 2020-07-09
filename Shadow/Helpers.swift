import Metal
import MetalKit
import Accelerate

/**
  Converts a CVPixelBuffer to an MTLTexture.
*/
public func CVPixelBufferToMTLTexture(pixelBuffer: CVPixelBuffer,
                                      textureCache: CVMetalTextureCache,
                                      pixelFormat: MTLPixelFormat = .bgra8Unorm) -> MTLTexture? {
  let width = CVPixelBufferGetWidth(pixelBuffer)
  let height = CVPixelBufferGetHeight(pixelBuffer)

  var texture: CVMetalTexture?
  let status = CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault,
    textureCache, pixelBuffer, nil, pixelFormat, width, height, 0, &texture)

  if status == kCVReturnSuccess, let texture = texture {
    return CVMetalTextureGetTexture(texture)
  }
  return nil
}

/**
  First crops the pixel buffer, then resizes it.
*/
public func resizePixelBuffer(_ srcPixelBuffer: CVPixelBuffer,
                              cropX: Int,
                              cropY: Int,
                              cropWidth: Int,
                              cropHeight: Int,
                              scaleWidth: Int,
                              scaleHeight: Int) -> CVPixelBuffer? {
  let flags = CVPixelBufferLockFlags(rawValue: 0)
  guard kCVReturnSuccess == CVPixelBufferLockBaseAddress(srcPixelBuffer, flags) else {
    return nil
  }
  defer { CVPixelBufferUnlockBaseAddress(srcPixelBuffer, flags) }

  guard let srcData = CVPixelBufferGetBaseAddress(srcPixelBuffer) else {
    print("Error: could not get pixel buffer base address")
    return nil
  }
  let srcBytesPerRow = CVPixelBufferGetBytesPerRow(srcPixelBuffer)
  let offset = cropY*srcBytesPerRow + cropX*4
  var srcBuffer = vImage_Buffer(data: srcData.advanced(by: offset),
                                height: vImagePixelCount(cropHeight),
                                width: vImagePixelCount(cropWidth),
                                rowBytes: srcBytesPerRow)

  let destBytesPerRow = scaleWidth*4
  guard let destData = malloc(scaleHeight*destBytesPerRow) else {
    print("Error: out of memory")
    return nil
  }
  var destBuffer = vImage_Buffer(data: destData,
                                 height: vImagePixelCount(scaleHeight),
                                 width: vImagePixelCount(scaleWidth),
                                 rowBytes: destBytesPerRow)

  let error = vImageScale_ARGB8888(&srcBuffer, &destBuffer, nil, vImage_Flags(0))
  if error != kvImageNoError {
    print("Error:", error)
    free(destData)
    return nil
  }

  let releaseCallback: CVPixelBufferReleaseBytesCallback = { _, ptr in
    if let ptr = ptr {
      free(UnsafeMutableRawPointer(mutating: ptr))
    }
  }

  let pixelFormat = CVPixelBufferGetPixelFormatType(srcPixelBuffer)
  var dstPixelBuffer: CVPixelBuffer?
  let status = CVPixelBufferCreateWithBytes(nil, scaleWidth, scaleHeight,
                                            pixelFormat, destData,
                                            destBytesPerRow, releaseCallback,
                                            nil, nil, &dstPixelBuffer)
  if status != kCVReturnSuccess {
    print("Error: could not create new pixel buffer")
    free(destData)
    return nil
  }
  return dstPixelBuffer
}

/**
  Resizes a CVPixelBuffer to a new width and height.
*/
public func resizePixelBuffer(_ pixelBuffer: CVPixelBuffer,
                              width: Int, height: Int) -> CVPixelBuffer? {
  return resizePixelBuffer(pixelBuffer, cropX: 0, cropY: 0,
                           cropWidth: CVPixelBufferGetWidth(pixelBuffer),
                           cropHeight: CVPixelBufferGetHeight(pixelBuffer),
                           scaleWidth: width, scaleHeight: height)
}

let textureLoader: MTKTextureLoader = {
  return MTKTextureLoader(device: MTLCreateSystemDefaultDevice()!)
}()

/**
  Loads a texture from the main bundle.
*/
public func loadTexture(named filename: String) -> MTLTexture? {
  if let url = Bundle.main.url(forResource: filename, withExtension: "") {
    return loadTexture(url: url)
  } else {
    print("Error: could not find image \(filename)")
    return nil
  }
}

/**
  Loads a texture from the specified URL.
*/
public func loadTexture(url: URL) -> MTLTexture? {
  do {
    return try textureLoader.newTexture(URL: url, options: [
      MTKTextureLoader.Option.SRGB : NSNumber(value: false)
    ])
  } catch {
    print("Error: could not load texture \(error)")
    return nil
  }
}

extension MTLComputeCommandEncoder {
  /**
    Dispatches a compute kernel on a 1-dimensional grid.

    - Parameters:
      - pipeline: the object with the compute function
      - count: the number of elements to process
  */
  public func dispatch(pipeline: MTLComputePipelineState, count: Int) {
    // Round off count to the nearest multiple of threadExecutionWidth.
    let width = pipeline.threadExecutionWidth
    let rounded = ((count + width - 1) / width) * width

    let blockSize = min(rounded, pipeline.maxTotalThreadsPerThreadgroup)
    let numBlocks = (count + blockSize - 1) / blockSize

    let threadGroupSize = MTLSizeMake(blockSize, 1, 1)
    let threadGroups = MTLSizeMake(numBlocks, 1, 1)

    setComputePipelineState(pipeline)
    dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
  }

  /**
    Dispatches a compute kernel on a 2-dimensional grid.

    - Parameters:
      - pipeline: the object with the compute function
      - width: the first dimension
      - height: the second dimension
  */
  public func dispatch(pipeline: MTLComputePipelineState, width: Int, height: Int) {
    let w = pipeline.threadExecutionWidth
    let h = pipeline.maxTotalThreadsPerThreadgroup / w

    let threadGroupSize = MTLSizeMake(w, h, 1)
    let threadGroups = MTLSizeMake(
      (width  + threadGroupSize.width  - 1) / threadGroupSize.width,
      (height + threadGroupSize.height - 1) / threadGroupSize.height, 1)

    setComputePipelineState(pipeline)
    dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
  }
}
