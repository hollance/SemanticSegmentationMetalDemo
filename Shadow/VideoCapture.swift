import UIKit
import AVFoundation
import CoreVideo

// TODO: This is not production-ready code!

public protocol VideoCaptureDelegate: class {
  func videoCapture(_ capture: VideoCapture, didCaptureSampleBuffer: CMSampleBuffer)
}

public class VideoCapture: NSObject {
  public var previewLayer: AVCaptureVideoPreviewLayer?
  public weak var delegate: VideoCaptureDelegate?

  let captureSession = AVCaptureSession()
  let videoOutput = AVCaptureVideoDataOutput()
  let queue = DispatchQueue(label: "net.machinethink.camera-queue")
  var orientation = AVCaptureVideoOrientation.portrait

  public func setUp(sessionPreset: AVCaptureSession.Preset = .high,
                    orientation: AVCaptureVideoOrientation = .portrait,
                    position: AVCaptureDevice.Position = .back,
                    completion: @escaping (Bool) -> Void) {
    self.orientation = orientation

    queue.async {
      let success = self.setUpCamera(sessionPreset: sessionPreset, position: position)
      DispatchQueue.main.async {
        completion(success)
      }
    }
  }

  func changeConfiguration(closure: () -> Bool) -> Bool {
    captureSession.beginConfiguration()
    let result = closure()
    videoOutput.connection(with: AVMediaType.video)?.videoOrientation = orientation
    captureSession.commitConfiguration()
    return result
  }

  func camera(withPosition position: AVCaptureDevice.Position) -> AVCaptureDevice? {
    return AVCaptureDevice.DiscoverySession(deviceTypes: [.builtInWideAngleCamera],
                                            mediaType: .video,
                                            position: position).devices.first
  }

  func setUpCamera(sessionPreset: AVCaptureSession.Preset,
                   position: AVCaptureDevice.Position) -> Bool {
    return changeConfiguration {
      captureSession.sessionPreset = sessionPreset

      guard let captureDevice = camera(withPosition: position) else {
        print("Error: no video devices available for position \(position.rawValue)")
        return false
      }

      // TODO: if no device available, maybe fallback to default?
      /*
      guard let captureDevice = AVCaptureDevice.default(for: AVMediaType.video) else {
        print("Error: no video devices available")
        return false
      }
      */

      guard let videoInput = try? AVCaptureDeviceInput(device: captureDevice) else {
        print("Error: could not create AVCaptureDeviceInput")
        return false
      }

      if captureSession.canAddInput(videoInput) {
        captureSession.addInput(videoInput)
      }

      let previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
      previewLayer.videoGravity = AVLayerVideoGravity.resizeAspect
      previewLayer.connection?.videoOrientation = orientation
      self.previewLayer = previewLayer

      // Don't mirror the front-facing camera in the preview so it looks the
      // same as the actual CVPixelBuffer. However, it might be nicer to flip
      // the texure as we draw it so that it *is* mirrored.
      previewLayer.connection?.automaticallyAdjustsVideoMirroring = false
      previewLayer.connection?.isVideoMirrored = false

      let settings: [String : Any] = [
        kCVPixelBufferPixelFormatTypeKey as String: NSNumber(value: kCVPixelFormatType_32BGRA)
      ]

      videoOutput.videoSettings = settings
      videoOutput.alwaysDiscardsLateVideoFrames = true
      videoOutput.setSampleBufferDelegate(self, queue: queue)
      if captureSession.canAddOutput(videoOutput) {
        captureSession.addOutput(videoOutput)
      }
      return true
    }
  }

  public func start() {
    if !captureSession.isRunning {
      captureSession.startRunning()
    }
  }

  public func stop() {
    if captureSession.isRunning {
      captureSession.stopRunning()
    }
  }

  public func switchCamera() {
    _ = changeConfiguration {
      if let currentInput = captureSession.inputs.first {
        var captureDevice: AVCaptureDevice?
        if (currentInput as! AVCaptureDeviceInput).device.position == .back {
          captureDevice = camera(withPosition: .front)
        } else {
          captureDevice = camera(withPosition: .back)
        }

        if let captureDevice = captureDevice {
          captureSession.removeInput(currentInput)
          if let videoInput = try? AVCaptureDeviceInput(device: captureDevice) {
            if captureSession.canAddInput(videoInput) {
              captureSession.addInput(videoInput)
            }
          } else {
            print("Error: could not create AVCaptureDeviceInput")
          }
        }
      }
      return true
    }
  }
}

extension VideoCapture: AVCaptureVideoDataOutputSampleBufferDelegate {
  public func captureOutput(_ output: AVCaptureOutput,
                            didOutput sampleBuffer: CMSampleBuffer,
                            from connection: AVCaptureConnection) {
    delegate?.videoCapture(self, didCaptureSampleBuffer: sampleBuffer)
  }

  public func captureOutput(_ output: AVCaptureOutput,
                            didDrop sampleBuffer: CMSampleBuffer,
                            from connection: AVCaptureConnection) {
    //print("dropped frame at", CMSampleBufferGetPresentationTimeStamp(sampleBuffer))
  }
}
