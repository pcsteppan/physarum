const PI = 3.1415926538;
const att_deposit = 5;

// Pixels
@group(0) @binding(0)  
  var<storage, read_write> pixels : array<vec4f>;

// Uniforms
@group(1) @binding(0) 
  var<uniform> rez : f32;

@group(1) @binding(1) 
  var<uniform> time : f32;

@group(1) @binding(2) 
  var<uniform> count : u32;

// Uniforms for physarum specifically
@group(1) @binding(3) 
  var<uniform> sensor_angle : f32;

@group(1) @binding(4) 
  var<uniform> sensor_distance : f32;

@group(1) @binding(5)  
  var<uniform> rotation_angle : f32;

@group(1) @binding(6)  
  var<uniform> velocity : f32;

// Other buffers
@group(2) @binding(0)  
  var<storage, read_write> positions : array<vec2f>;

@group(2) @binding(1)  
  var<storage, read_write> headings : array<vec2f>;

@group(2) @binding(2)  
  var<storage, read_write> attGrid : array<f32>;

fn r(n: f32) -> f32 {
  let x = sin(n) * 43758.5453;
  return fract(x);
}

fn index(p: vec2f) -> i32 {
  return i32(p.x) + i32(p.y) * i32(rez);
}

@compute @workgroup_size(256)
fn reset(@builtin(global_invocation_id) id : vec3u) {
  let seed = f32(id.x)/f32(count);
  var p = vec2(r(seed), r(seed + 0.1));
  p *= rez;
  positions[id.x] = p;
  headings[id.x] = normalize(vec2(r(f32(id.x+1)), r(f32(id.x + 2))) - 0.5);
}

@compute @workgroup_size(256)
fn simulate(@builtin(global_invocation_id) id : vec3u) {
  var p = positions[id.x];
  var h = headings[id.x];

  var norm_h = normalize(h);

  // sensor positions
  var p_f = wrap(p + sensor_distance * norm_h);
  var p_fl = wrap(p + sensor_distance * rotate2d(sensor_angle) * norm_h);
  var p_fr = wrap(p + sensor_distance * rotate2d(-sensor_angle) * norm_h);

  // sensor values
  var f = attGrid[index(p_f)];
  var fl = attGrid[index(p_fl)];
  var fr = attGrid[index(p_fr)];

  var rotation_sign = 0;
  if (f > fl && f > fr) 
  {
    // do nothing
  } 
  else if (f < fl && f < fr) 
  {
    rotation_sign = select(-1, 1, r(f32(id.x) / f32(count) + time) > 0.5);
  } 
  else if (fl < fr) 
  {
    rotation_sign = -1;
  } 
  else if (fl > fr) 
  {
    rotation_sign = 1;
  }

  h = rotate2d(f32(rotation_sign) * rotation_angle) * norm_h;
  headings[id.x] = h;

  p = positions[id.x];
  p += velocity * h;
  positions[id.x] = wrap(p);

  // AGENT RENDERING
  var hue = (f32(rotation_sign + 1) + .3) % 1;
  pixels[index(p)] += vec4(hsl_to_rgb(vec3(hue, 1., .55)), 1.);

  attGrid[index(p)] += att_deposit;
}

@compute @workgroup_size(256)
fn render(@builtin(global_invocation_id) id : vec3u) 
{
  var p = positions[id.x];

  // ATT GRID RENDERING
  pixels[id.x] += vec4(0., 0., max(0, attGrid[id.x]) / 5., 1.);
  pixels[id.x] *= 0.3;

  attGrid[id.x] *= .91;
}

@compute @workgroup_size(16, 16)
fn diffuse(@builtin(global_invocation_id) id : vec3u) 
{
  var p = vec2(f32(id.x), f32(id.y));
  var sum = 0.0;
  
  var kernelSize = 1.;

  for(var x = -kernelSize; x <= kernelSize; x += 1.0) 
  {
    for(var y = -kernelSize; y <= kernelSize; y += 1.0) 
    {
      sum += attGrid[index(p + vec2(x, y))];
    }
  }

  var norm_p = p / rez;
  var close_to_edgeness_x = -pow((norm_p.x - .5) * 2, 4) + 1;
  var close_to_edgeness_y = -pow((norm_p.y - .5) * 2, 4) + 1;
  attGrid[index(p)] *= min(close_to_edgeness_x, close_to_edgeness_y);
}

fn rotate2d(angle : f32) -> mat2x2<f32> 
{
    return mat2x2<f32>(cos(angle), -sin(angle),
                       sin(angle),  cos(angle));
}

fn hsl_to_rgb(hsl: vec3f) -> vec3f
{
   let r = abs(hsl.x * 6.0 - 3.0) - 1.0;
   let g = 2.0 - abs(hsl.x * 6.0 - 2.0);
   let b = 2.0 - abs(hsl.x * 6.0 - 4.0);
   let c = (1.0 - abs(2.0 * hsl.z - 1.0)) * hsl.y;
   var rgb = vec3(r, g, b);
   return vec3((rgb - 0.5) * c + hsl.z);
}

fn wrap(position: vec2f) -> vec2f
{
  return (position + rez) % rez;
} 