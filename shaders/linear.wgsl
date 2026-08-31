struct Parameters {
  rows: u32,
  columns: u32,
  batch: u32,
  padding: u32,
}

@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> activations: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> parameters: Parameters;

@compute @workgroup_size(8, 8)
fn linear(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x;
  let batch_index = id.y;
  if (row >= parameters.rows || batch_index >= parameters.batch) { return; }
  var sum = 0.0;
  for (var column = 0u; column < parameters.columns; column += 1u) {
    sum += weights[row * parameters.columns + column] *
           activations[batch_index * parameters.columns + column];
  }
  output[batch_index * parameters.rows + row] = sum;
}
