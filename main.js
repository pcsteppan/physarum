import { createShader, render } from "./lib.js";

const agentWorkgroupsDenominator = 256;

const size = (type, numBytes) => { return { type, numBytes } }
const sizes = {
	f32: size('f32', 4),
	u32: size('u32', 4),
	i32: size('i32', 4),
	vec2: size('vec2', 8),
	vec4: size('vec4', 16),
};

const uniform = (value, size) => { return { value, size } }
const uniforms = {
	rez: uniform(2048, sizes.f32),
	time: uniform(0, sizes.f32),
	count: uniform(2000000, sizes.u32),
	sensor_angle: uniform(Math.PI / 6., sizes.f32),
	sensor_distance: uniform(12, sizes.f32),
	rotation_angle: uniform(Math.PI / 32., sizes.f32),
	velocity: uniform(.7, sizes.f32)
};

const presets = {
	DEFAULT: {
		sensor_angle: uniforms.sensor_angle.value,
		sensor_distance: uniforms.sensor_distance.value,
		rotation_angle: uniforms.rotation_angle.value,
		velocity: uniforms.velocity.value,
	},
	MICRO: {
		sensor_angle: .52,
		sensor_distance: 10,
		rotation_angle: .13,
		velocity: 1.5,
	},
	LACE: {
		sensor_angle: .35,
		sensor_distance: 35,
		rotation_angle: .13,
		velocity: 1.25,
	},
	UNDULATE: {
		sensor_angle: .63,
		sensor_distance: 18.5,
		rotation_angle: 0.9,
		velocity: 8.6
	},
	AURA: {
		sensor_angle: 1.3,
		sensor_distance: 190,
		rotation_angle: 0.095,
		velocity: 9.9
	}
}

let selectedPreset = {
	current: 'SELECT A PRESET'
}

const settings = {
	scale:
		(0.95 * Math.min(window.innerHeight, window.innerWidth)) / uniforms.rez.value,
	pixelWorkgroups: Math.ceil(uniforms.rez.value ** 2 / 256),
	agentWorkgroups: Math.ceil(uniforms.count.value / agentWorkgroupsDenominator),
	gridWorkgroups: Math.ceil(uniforms.rez.value / 16), // the ^2 on each side factors out
};

async function main() {
	if (!navigator.gpu) {
		throw Error("WebGPU not supported.");
	}

	const adapter = await navigator.gpu.requestAdapter();
	if (!adapter) {
		throw Error("Couldn't request WebGPU adapter.");
	}

	const gpu = await adapter.requestDevice();

	const canvas = document.createElement("canvas");
	canvas.width = canvas.height = uniforms.rez.value * settings.scale;
	document.body.appendChild(canvas);
	const context = canvas.getContext("webgpu");
	const format = "bgra8unorm";
	context.configure({
		device: gpu,
		format: format,
		alphaMode: "premultiplied",
	});

	const visibility = GPUShaderStage.COMPUTE;

	const pixelBuffer = gpu.createBuffer({
		size: uniforms.rez.value ** 2 * sizes.vec4.numBytes,
		usage: GPUBufferUsage.STORAGE,
	});
	const pixelBufferLayout = gpu.createBindGroupLayout({
		entries: [{ visibility, binding: 0, buffer: { type: "storage" } }],
	});
	const pixelBufferBindGroup = gpu.createBindGroup({
		layout: pixelBufferLayout,
		entries: [{ binding: 0, resource: { buffer: pixelBuffer } }],
	});

	const uniformBuffers = {};

	for (let [k, v] of Object.entries(uniforms)) {
		uniformBuffers[k] = gpu.createBuffer({
			size: v.size.numBytes,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
		});
	}

	const uniformsLayoutEntries = new Array(Object.keys(uniforms).length).fill(null).map((_, i) => {
		return {
			visibility, binding: i, buffer: { type: 'uniform' }
		}
	});
	const uniformsLayout = gpu.createBindGroupLayout({
		entries: uniformsLayoutEntries
	});

	const uniformBuffersBindGroup = gpu.createBindGroup({
		layout: uniformsLayout,
		entries: Array.from(Object.keys(uniforms)).map((k, i) => {
			return {
				binding: i,
				resource: {
					buffer: uniformBuffers[k]
				}
			}
		})
	});

	const writeUniforms = () => {
		for (let [k, v] of Object.entries(uniformBuffers)) {
			const newValue = uniforms[k].value;
			const newBufferValue = Array.isArray(newValue) ? newValue : [newValue];
			const newBuffer = uniforms[k].size.type === sizes.u32.type
				? new Uint32Array(newBufferValue)
				: new Float32Array(newBufferValue);
			gpu.queue.writeBuffer(v, 0, newBuffer);
		}
		settings.agentWorkgroups = Math.ceil(uniforms.count.value / agentWorkgroupsDenominator);
	};

	writeUniforms();

	const positionsBuffer = gpu.createBuffer({
		size: sizes.vec2.numBytes * uniforms.count.value,
		usage: GPUBufferUsage.STORAGE,
	});

	const attGridBuffer = gpu.createBuffer({
		size: sizes.f32.numBytes * uniforms.rez.value ** 2,
		usage: GPUBufferUsage.STORAGE,
	});

	const headingsBuffer = gpu.createBuffer({
		size: sizes.vec2.numBytes * uniforms.count.value,
		usage: GPUBufferUsage.STORAGE,
	});

	const agentsLayout = gpu.createBindGroupLayout({
		entries: [
			{ visibility, binding: 0, buffer: { type: "storage" } },
			{ visibility, binding: 1, buffer: { type: "storage" } },
			{ visibility, binding: 2, buffer: { type: "storage" } },
		],
	});

	const agentsBuffersBindGroup = gpu.createBindGroup({
		layout: agentsLayout,
		entries: [
			{ binding: 0, resource: { buffer: positionsBuffer } },
			{ binding: 1, resource: { buffer: headingsBuffer } },
			{ binding: 2, resource: { buffer: attGridBuffer } },
		],
	});

	const layout = gpu.createPipelineLayout({
		bindGroupLayouts: [pixelBufferLayout, uniformsLayout, agentsLayout],
	});

	const module = await createShader(gpu, "compute.wgsl");

	const resetPipeline = gpu.createComputePipeline({
		layout,
		compute: { module, entryPoint: "reset" },
	});

	const simulatePipeline = gpu.createComputePipeline({
		layout,
		compute: { module, entryPoint: "simulate" },
	});

	const renderPipeline = gpu.createComputePipeline({
		layout,
		compute: { module, entryPoint: "render" },
	});

	const diffusePipeline = gpu.createComputePipeline({
		layout,
		compute: { module, entryPoint: "diffuse" },
	});

	const reset = () => {
		const encoder = gpu.createCommandEncoder();
		const pass = encoder.beginComputePass();
		pass.setPipeline(resetPipeline);
		pass.setBindGroup(0, pixelBufferBindGroup);
		pass.setBindGroup(1, uniformBuffersBindGroup);
		pass.setBindGroup(2, agentsBuffersBindGroup);
		pass.dispatchWorkgroups(settings.agentWorkgroups);
		pass.end();
		gpu.queue.submit([encoder.finish()]);
	};
	reset();

	const draw = () => {
		const encoder = gpu.createCommandEncoder();
		const pass = encoder.beginComputePass();
		pass.setBindGroup(0, pixelBufferBindGroup);
		pass.setBindGroup(1, uniformBuffersBindGroup);
		pass.setBindGroup(2, agentsBuffersBindGroup);

		pass.setPipeline(diffusePipeline);
		pass.dispatchWorkgroups(settings.gridWorkgroups, settings.gridWorkgroups);

		pass.setPipeline(simulatePipeline);
		pass.dispatchWorkgroups(settings.agentWorkgroups);

		pass.setPipeline(renderPipeline);
		pass.dispatchWorkgroups(settings.pixelWorkgroups);

		pass.end();

		render(gpu, uniforms.rez.value, pixelBuffer, format, context, encoder);
		gpu.queue.submit([encoder.finish()]);
		gpu.queue.writeBuffer(uniformBuffers.time, 0, new Float32Array([uniforms.time.value++]));
		requestAnimationFrame(draw);
		// setTimeout(() => draw(), 10);
	};
	draw();

	let gui = new lil.GUI();
	gui.add(uniforms.sensor_angle, 'value', 0, Math.PI / 2.)
		.name('sensor_angle')
		.listen();
	gui.add(uniforms.sensor_distance, 'value', 0, 1024)
		.name('sensor_distance')
		.listen();
	gui.add(uniforms.rotation_angle, 'value', 0.0, Math.PI / 2.)
		.name('rotation_angle')
		.listen();
	gui.add(uniforms.count, 'value', 0, 3000000)
		.name('count')
		.listen();
	gui.add(uniforms.velocity, 'value', 0, 15)
		.name('velocity')
		.listen();
	gui.add(selectedPreset, 'current', Object.keys(presets)).onChange(preset => {
		selectedPreset.current = preset;
		const newSelectedPreset = presets[preset];
		for (let k in newSelectedPreset) {
			uniforms[k].value = newSelectedPreset[k];
		}
		writeUniforms();
	});
	gui.onChange(writeUniforms);
}
main();

