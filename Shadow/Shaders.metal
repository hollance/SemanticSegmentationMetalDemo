#include <metal_stdlib>
using namespace metal;

struct VertexOut {
  float4 position [[position]];
  float2 texCoord;
};

vertex VertexOut vertexFunc(uint vid [[vertex_id]]) {
  constexpr float2 pos[] = {
    float2(-1.0f, 1.0f), float2(-1.0f, -1.0f),
    float2( 1.0f, 1.0f), float2( 1.0f, -1.0f)
  };

  constexpr float2 uv[] = {
    float2( 0.0f, 0.0f), float2( 0.0f, 1.0f),
    float2( 1.0f, 0.0f), float2( 1.0f, 1.0f)
  };

  VertexOut out;
  out.position = float4(pos[vid], 0.0, 1.0);
  out.texCoord = uv[vid];
  return out;
}

fragment half4 fragmentFunc(
  VertexOut in [[stage_in]],
  texture2d<float, access::sample> tex2d [[texture(0)]])
{
  constexpr sampler s(coord::normalized,
                      address::clamp_to_zero,
                      filter::linear);

  const half4 color = half4(tex2d.sample(s, in.texCoord));
  return color;
}

struct MixParams {
  int segmentationWidth;
  int segmentationHeight;
  float dx;
  float dy;
};

/*
  Converts sigmoid output into argmax output.
*/
kernel void convertProbabilitiesToMask(
  device float* probabilitiesMask [[buffer(0)]],
  device int* segmentationMask [[buffer(1)]],
  constant MixParams& params [[buffer(2)]],
  uint gid [[thread_position_in_grid]])
{
  if (gid >= (uint)(params.segmentationHeight * params.segmentationWidth)) return;

  const auto prob = probabilitiesMask[gid];
  if (prob > 0.5f) {
    segmentationMask[gid] = 15;  // person
  } else {
    segmentationMask[gid] = 0;
  }
}

/*
  Returns the predicted class label at the specified pixel coordinate.
  The position should be normalized between 0 and 1.
*/
static inline int get_class(float2 pos, int width, int height, device int* mask) {
  const int x = int(pos.x * width);
  const int y = int(pos.y * height);
  return mask[y*width + x];
}

/*
  Returns the probability that the specified pixel coordinate contains the
  class "person". The position should be normalized between 0 and 1.
*/
static float get_person_probability(float2 pos, int width, int height, device int* mask) {
  // TODO: We map the full-size texture position to a location in the smaller
  // segmentation mask. This essentially creates nearest-neighbor upsampling,
  // causing jaggies. Perhaps we should not ignore the fractional part and do
  // a blend of the surrounding pixels to get a probability, i.e. this pixel
  // is 75% person and 25% background.

  return get_class(pos, width, height, mask) == 15;
}

/*
  Copies the person probabilities form the segmentation mask MTLBuffer into
  a texture with 1 color channel.
*/
kernel void convertMaskToTexture(
  texture2d<float, access::write> texture [[texture(0)]],
  device int* segmentationMask [[buffer(0)]],
  uint2 gid [[thread_position_in_grid]])
{
  if (gid.x >= texture.get_width() ||
      gid.y >= texture.get_height()) return;

  const float2 pos = float2(float(gid.x) / float(texture.get_width()),
                            float(gid.y) / float(texture.get_height()));
  const float is_person = get_person_probability(pos, texture.get_width(),
                                                 texture.get_height(),
                                                 segmentationMask);
  texture.write(is_person, gid);
}

/*
  Draws the segmentation mask. Each class gets a different color.
  dx determines how much the mask is blended with the original input.
*/
kernel void maskColors(
  texture2d<float, access::read> inputTexture [[texture(0)]],
  texture2d<float, access::write> outputTexture [[texture(1)]],
  device int* segmentationMask [[buffer(0)]],
  constant MixParams& params [[buffer(1)]],
  device unsigned char* colors [[buffer(2)]],
  uint2 gid [[thread_position_in_grid]])
{
  if (gid.x >= outputTexture.get_width() ||
      gid.y >= outputTexture.get_height()) return;

  const float2 pos = float2(float(gid.x) / float(outputTexture.get_width()),
                            float(gid.y) / float(outputTexture.get_height()));
  const int value = get_class(pos, params.segmentationWidth,
                              params.segmentationHeight, segmentationMask);

  const float4 color = float4(colors[value*3 + 0] / 255.0f,
                              colors[value*3 + 1] / 255.0f,
                              colors[value*3 + 2] / 255.0f,
                              1.0f);

  const float blend = (params.dx + 1.0f) / 2.0f;
  const float4 inPixel = float4(inputTexture.read(gid));
  const float4 outPixel = mix(color, inPixel, blend);

  outputTexture.write(outPixel, gid);
}

/*
  Replaces the background and all non-person classes with a different
  background. Also draws a shadow behind the person.
*/
kernel void shadow(
  texture2d<float, access::read> inputTexture [[texture(0)]],
  texture2d<float, access::write> outputTexture [[texture(1)]],
  texture2d<float, access::sample> backgroundTexture [[texture(2)]],
  device int* segmentationMask [[buffer(0)]],
  constant MixParams& params [[buffer(1)]],
  uint2 gid [[thread_position_in_grid]])
{
  if (gid.x >= inputTexture.get_width() ||
      gid.y >= inputTexture.get_height()) return;

  constexpr sampler s(coord::normalized,
                      address::clamp_to_zero,
                      filter::linear);

  const float2 pos = float2(float(gid.x) / float(outputTexture.get_width()),
                            float(gid.y) / float(outputTexture.get_height()));
  const float is_person = get_person_probability(pos, params.segmentationWidth,
                                                 params.segmentationHeight,
                                                 segmentationMask);

  const float4 inPixel = inputTexture.read(gid);
  float4 outPixel = inPixel;

  if (is_person < 0.5f) {
    // Use a sampler so that the background texture doesn't have to be the
    // same size as the input texture.
    outPixel = backgroundTexture.sample(s, float2(float(gid.x) / float(inputTexture.get_width()),
                                                  float(gid.y) / float(inputTexture.get_height())));

    // If the pixel at offset (dx, dy) is a person pixel, then we're a shadow.
    const int dx = int(params.dx * 200);
    const int dy = int(params.dy * 200);
    const int ox = gid.x + dx;
    const int oy = gid.y + dy;
    if (ox >= 0 && ox < (int)inputTexture.get_width() &&
        oy >= 0 && oy < (int)inputTexture.get_height()) {

      const float2 pos = float2(float(ox) / float(outputTexture.get_width()),
                                float(oy) / float(outputTexture.get_height()));
      const float is_person = get_person_probability(pos, params.segmentationWidth,
                                                     params.segmentationHeight,
                                                     segmentationMask);
      if (is_person >= 0.5f) {
        outPixel = mix(float4(0.0f, 0.0f, 0.0f, 1.0f), outPixel, 0.7f);
      }
    }
  }

  outputTexture.write(outPixel, gid);
}

// Based on code from http://lolengine.net/blog/2013/07/27/rgb-to-hsv-in-glsl
inline float3 rgb2hsv(float3 c) {
  constexpr float4 K = float4(0.0f, -1.0f / 3.0f, 2.0f / 3.0f, -1.0f);
  float4 p = mix(float4(c.bg, K.wz), float4(c.gb, K.xy), step(c.b, c.g));
  float4 q = mix(float4(p.xyw, c.r), float4(c.r, p.yzx), step(p.x, c.r));
  float d = q.x - min(q.w, q.y);
  constexpr float e = 1.0e-10f;
  return float3(abs(q.z + (q.w - q.y) / (6.0f * d + e)), d / (q.x + e), q.x);
}

// Based on code from http://lolengine.net/blog/2013/07/27/rgb-to-hsv-in-glsl
inline float3 hsv2rgb(float3 c) {
  constexpr float4 K = float4(1.0f, 2.0f / 3.0f, 1.0f / 3.0f, 3.0f);
  float3 p = abs(fract(c.xxx + K.xyz) * 6.0f - K.www);
  return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0f, 1.0f), c.y);
}

/*
  Changes the saturation and brightness for background pixels.
*/
kernel void saturation(
  texture2d<float, access::read> inputTexture [[texture(0)]],
  texture2d<float, access::write> outputTexture [[texture(1)]],
  device int* segmentationMask [[buffer(0)]],
  constant MixParams& params [[buffer(1)]],
  uint2 gid [[thread_position_in_grid]])
{
  if (gid.x >= outputTexture.get_width() ||
      gid.y >= outputTexture.get_height()) return;

  const float2 pos = float2(float(gid.x) / float(outputTexture.get_width()),
                            float(gid.y) / float(outputTexture.get_height()));
  const float is_person = get_person_probability(pos, params.segmentationWidth,
                                                 params.segmentationHeight,
                                                 segmentationMask);

  float4 pixel = float4(inputTexture.read(gid));

  if (is_person < 0.5f) {
    float3 hsv = rgb2hsv(pixel.rgb);
    hsv.y *= (params.dx + 1.0f);
    hsv.z *= (params.dy + 1.0f);
    float3 rgb = clamp(hsv2rgb(hsv), 0.0f, 1.0f);
    pixel = float4(rgb, 1.0f);
  }

  outputTexture.write(pixel, gid);
}

/*
  Rounds down the position where we read the pixel.
*/
kernel void pixelate(
  texture2d<float, access::read> inputTexture [[texture(0)]],
  texture2d<float, access::write> outputTexture [[texture(1)]],
  device int* segmentationMask [[buffer(0)]],
  constant MixParams& params [[buffer(1)]],
  uint2 gid [[thread_position_in_grid]])
{
  if (gid.x >= outputTexture.get_width() ||
      gid.y >= outputTexture.get_height()) return;

  const float2 pos = float2(float(gid.x) / float(outputTexture.get_width()),
                            float(gid.y) / float(outputTexture.get_height()));
  const float is_person = get_person_probability(pos, params.segmentationWidth,
                                                 params.segmentationHeight,
                                                 segmentationMask);

  int coarseness;
  if (is_person < 0.5f) {
    coarseness = max(1, int((params.dx + 1.0f) * 16.0f));
  } else {
    coarseness = max(1, int((params.dy + 1.0f) * 16.0f));
  }

  uint2 readPos = (gid.xy / coarseness) * coarseness /*+ coarseness/2*/;
  float4 pixel = float4(inputTexture.read(readPos));
  outputTexture.write(pixel, gid);
}

/*
  Replaces the background and all non-person classes with a different
  background.
*/
kernel void composite(
  texture2d<float, access::read> inputTexture [[texture(0)]],
  texture2d<float, access::write> outputTexture [[texture(1)]],
  texture2d<float, access::sample> backgroundTexture [[texture(2)]],
  device int* segmentationMask [[buffer(0)]],
  constant MixParams& params [[buffer(1)]],
  uint2 gid [[thread_position_in_grid]])
{
  if (gid.x >= inputTexture.get_width() ||
      gid.y >= inputTexture.get_height()) return;

  constexpr sampler s(coord::normalized,
                      address::clamp_to_zero,
                      filter::linear);

  const float2 pos = float2(float(gid.x) / float(outputTexture.get_width()),
                            float(gid.y) / float(outputTexture.get_height()));
  const float is_person = get_person_probability(pos, params.segmentationWidth,
                                                 params.segmentationHeight,
                                                 segmentationMask);

  const float4 inPixel = inputTexture.read(gid);
  float4 outPixel = inPixel;

  if (is_person < 0.5f) {
    // Use a sampler so that the background texture doesn't have to be the
    // same size as the input texture.
    outPixel = backgroundTexture.sample(s, float2(float(gid.x) / float(inputTexture.get_width()),
                                                  float(gid.y) / float(inputTexture.get_height())));
  }

  outputTexture.write(outPixel, gid);
}

/*
  Uses the glowTexture as the blending factor for blending a purple color
  with the inputTexture, except where the mask is class "person".
*/
kernel void glow(
  texture2d<float, access::read> inputTexture [[texture(0)]],
  texture2d<float, access::write> outputTexture [[texture(1)]],
  texture2d<float, access::sample> glowTexture [[texture(2)]],
  device int* segmentationMask [[buffer(0)]],
  constant MixParams& params [[buffer(1)]],
  uint2 gid [[thread_position_in_grid]])
{
  if (gid.x >= inputTexture.get_width() ||
      gid.y >= inputTexture.get_height()) return;

  constexpr sampler s(coord::normalized,
                      address::clamp_to_zero,
                      filter::linear);

  const float2 pos = float2(float(gid.x) / float(outputTexture.get_width()),
                            float(gid.y) / float(outputTexture.get_height()));
  const float is_person = get_person_probability(pos, params.segmentationWidth,
                                                 params.segmentationHeight,
                                                 segmentationMask);

  const float4 inPixel = inputTexture.read(gid);
  float4 outPixel = inPixel;

  if (is_person < 0.5f) {
    float4 glow = glowTexture.sample(s, pos);
    outPixel = mix(inPixel, float4(1.0f, 0.3f, 0.8f, 1.0f), glow.r);
  }

  outputTexture.write(outPixel, gid);
}
