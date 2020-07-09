/* Copyright © 2016-2018 M.I. Hollemans. All rights reserved. */

import Foundation
import QuartzCore

/**
  Simple frames-per-second counter for processing video frames.
*/
public class FPSCounter {
  private(set) public var fps: Double = 0

  var frames = 0
  var startTime: CFTimeInterval = 0

  public init() { }

  public func start() {
    frames = 0
    startTime = CACurrentMediaTime()
  }

  /*
    Call this after completing work on a video frame. It updates the `fps`
    variable by counting how many frames were completed in the last second.
  */
  public func frameCompleted() {
    frames += 1
    let now = CACurrentMediaTime()
    let elapsed = now - startTime
    if elapsed >= 0.01 {
      let current = Double(frames) / elapsed
      let smoothing = 0.75
      fps = smoothing*fps + (1 - smoothing)*current
      if elapsed >= 1 {
        frames = 0
        startTime = now
      }
    }
  }
}
