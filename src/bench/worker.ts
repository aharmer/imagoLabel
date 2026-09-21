// Runs one SAM model off the main thread: load → prepare image → encode → decode clicks.
// The page creates a fresh worker per benchmark so each model starts with clean memory.
import { AutoModel, AutoProcessor, RawImage, Tensor, env } from '@huggingface/transformers';
import { bestMask, keepBestRegion, upsampleMask } from '../shared/mask';
import { configureOnnxRuntime, onnxEnv } from '../shared/ort';
import type { BenchResult, Box, FromWorker, MaskData, ModelConfig, Point, ToWorker } from './protocol';

interface WorkerScope {
  postMessage(message: FromWorker, transfer?: Transferable[]): void;
  onmessage: ((event: MessageEvent<ToWorker>) => void) | null;
}
const scope = self as unknown as WorkerScope;
const post = (message: FromWorker, transfer: Transferable[] = []) => scope.postMessage(message, transfer);

configureOnnxRuntime();
const onnx = onnxEnv();

const DTYPE_SUFFIX: Record<ModelConfig['dtype'], string> = { fp32: '', fp16: '_fp16', q8: '_quantized', q4f16: '_q4f16' };
const MAX_FULL_RES_PIXELS = 120_000_000;

// Transformers.js model and processor types are too loose to be useful here.
let model: any = null;
let processor: any = null;
let current: {
  embeddings: Record<string, Tensor>;
  inputs: any;
  scale: number;
  width: number;
  height: number;
} | null = null;

const median = (xs: number[]) => {
  if (xs.length === 0) return null;
  const s = [...xs].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
};

async function timed<T>(fn: () => Promise<T>): Promise<[T, number]> {
  const start = performance.now();
  const value = await fn();
  return [value, performance.now() - start];
}

async function isCached(config: ModelConfig) {
  try {
    const cache = await caches.open(env.cacheKey ?? 'transformers-cache');
    const url = `https://huggingface.co/${config.repo}/resolve/main/onnx/vision_encoder${DTYPE_SUFFIX[config.dtype]}.onnx`;
    return Boolean(await cache.match(url));
  } catch {
    return false;
  }
}

async function loadModel(config: ModelConfig) {
  const files = new Map<string, { loaded: number; total: number }>();
  let lastPost = 0;
  const progress_callback = (p: any) => {
    if (p.status !== 'progress' || !p.file) return;
    files.set(p.file, { loaded: p.loaded ?? 0, total: p.total ?? 0 });
    const now = performance.now();
    if (now - lastPost < 150) return;
    lastPost = now;
    let loadedBytes = 0;
    let totalBytes = 0;
    for (const f of files.values()) {
      loadedBytes += f.loaded;
      totalBytes += f.total;
    }
    post({ type: 'progress', stage: 'loading', loadedBytes, totalBytes });
  };
  processor = await AutoProcessor.from_pretrained(config.repo, { progress_callback });
  model = await AutoModel.from_pretrained(config.repo, {
    device: { vision_encoder: config.device, prompt_encoder_mask_decoder: 'wasm' },
    dtype: { vision_encoder: config.dtype, prompt_encoder_mask_decoder: config.decoderDtype },
    progress_callback,
  });
}

/** Decode the image and downscale it natively to ≤1024 px before handing it to the model's processor. */
async function prepareImage(blob: Blob) {
  const [full, decodeMs] = await timed(() => createImageBitmap(blob));
  const [prepared, prepMs] = await timed(async () => {
    const { width, height } = full;
    const scale = Math.min(1, 1024 / Math.max(width, height));
    const w = Math.round(width * scale);
    const h = Math.round(height * scale);
    const small = scale < 1 ? await createImageBitmap(full, { resizeWidth: w, resizeHeight: h, resizeQuality: 'high' }) : full;
    const ctx = new OffscreenCanvas(w, h).getContext('2d')!;
    ctx.drawImage(small, 0, 0);
    if (small !== full) small.close();
    full.close();
    const raw = new RawImage(ctx.getImageData(0, 0, w, h).data, w, h, 4).rgb();
    const inputs = await processor(raw);
    return { inputs, scale, width, height };
  });
  return { ...prepared, decodeMs, prepMs };
}

async function runDecoder(points: Point[]) {
  if (!current) throw new Error('No image has been encoded yet.');
  const { inputs, scale } = current;
  const scaled = points.map((p) => [p.x * scale, p.y * scale]);
  const input_points = processor.reshape_input_points([[scaled]], inputs.original_sizes, inputs.reshaped_input_sizes);
  const labels = BigInt64Array.from(points.map((p) => (p.positive ? 1n : 0n)));
  const input_labels = new Tensor('int64', labels, [1, 1, points.length]);
  return model({ ...current.embeddings, input_points, input_labels });
}

/** Pick the best mask for the current image. */
function pickMask(outputs: any) {
  if (!current) throw new Error('No image has been encoded yet.');
  const ip = processor.image_processor;
  return bestMask(outputs, current.inputs.reshaped_input_sizes[0], ip.do_pad && ip.pad_size ? ip.pad_size : null);
}

function renderMask(outputs: any, display: { width: number; height: number }, points: Point[]): MaskData {
  const m = pickMask(outputs);
  const { width, height } = display;
  const mask = upsampleMask(m, width, height);
  const toDisplay = width / current!.width;
  keepBestRegion(mask, width, height, points.filter((p) => p.positive).map((p) => [p.x * toDisplay, p.y * toDisplay]));
  const rgba = new Uint8ClampedArray(width * height * 4);
  let minX = width, minY = height, maxX = -1, maxY = -1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (!mask[y * width + x]) continue;
      const i = (y * width + x) * 4;
      rgba[i] = 30;
      rgba[i + 1] = 144;
      rgba[i + 2] = 255;
      rgba[i + 3] = 115;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
  }
  const sx = current!.width / width;
  const sy = current!.height / height;
  const box: Box | null =
    maxX < 0 ? null : { x: minX * sx, y: minY * sy, width: (maxX - minX + 1) * sx, height: (maxY - minY + 1) * sy };
  return { pixels: new ImageData(rgba, width, height), box, score: m.score };
}

async function benchmark(msg: Extract<ToWorker, { type: 'benchmark' }>) {
  if (msg.numThreads !== 'auto') onnx.wasm.numThreads = msg.numThreads;
  const cachedBeforeRun = await isCached(msg.config);

  post({ type: 'progress', stage: 'loading' });
  const [, loadMs] = await timed(() => loadModel(msg.config));

  post({ type: 'progress', stage: 'image' });
  const image = await prepareImage(msg.image);

  const encodeTimes: number[] = [];
  for (let i = 0; i < msg.encodeRuns; i++) {
    post({ type: 'progress', stage: 'encoding', step: i + 1, steps: msg.encodeRuns });
    const [embeddings, ms] = await timed(() => model.get_image_embeddings(image.inputs));
    current = { embeddings: embeddings as Record<string, Tensor>, inputs: image.inputs, scale: image.scale, width: image.width, height: image.height };
    encodeTimes.push(ms);
  }

  const points = msg.points.length ? msg.points : [{ x: image.width / 2, y: image.height / 2, positive: true }];
  const decodeTimes: number[] = [];
  let outputs: any = null;
  for (let i = 0; i < msg.decodeRuns; i++) {
    post({ type: 'progress', stage: 'decoding', step: i + 1, steps: msg.decodeRuns });
    const [out, ms] = await timed(() => runDecoder(points));
    outputs = out;
    decodeTimes.push(ms);
  }

  let fullResMaskMs: number | null = null;
  if (msg.measureFullRes && image.width * image.height <= MAX_FULL_RES_PIXELS) {
    post({ type: 'progress', stage: 'full-res mask' });
    const m = pickMask(outputs);
    const start = performance.now();
    upsampleMask(m, image.width, image.height);
    fullResMaskMs = performance.now() - start;
  }

  const mask = renderMask(outputs, msg.display, points);
  const result: BenchResult = {
    key: msg.config.key,
    numThreads: msg.config.device === 'wasm' ? (onnx.wasm.numThreads ?? msg.numThreads) : msg.numThreads,
    cachedBeforeRun,
    loadMs,
    imageWidth: image.width,
    imageHeight: image.height,
    imageDecodeMs: image.decodeMs,
    imagePrepMs: image.prepMs,
    encodeFirstMs: encodeTimes[0],
    encodeMedianMs: median(encodeTimes.slice(1)),
    decodeFirstMs: decodeTimes[0],
    decodeMedianMs: median(decodeTimes.slice(1)),
    fullResMaskMs,
    score: mask.score,
  };
  post({ type: 'result', result, mask }, [mask.pixels.data.buffer]);
}

async function decode(msg: Extract<ToWorker, { type: 'decode' }>) {
  const [outputs, ms] = await timed(() => runDecoder(msg.points));
  const mask = renderMask(outputs, msg.display, msg.points);
  post({ type: 'decoded', mask, ms }, [mask.pixels.data.buffer]);
}

// Handle messages one at a time so rapid clicks queue rather than overlap.
let queue = Promise.resolve();
scope.onmessage = (event) => {
  const msg = event.data;
  queue = queue.then(async () => {
    try {
      if (msg.type === 'benchmark') await benchmark(msg);
      else await decode(msg);
    } catch (err) {
      post({ type: 'error', message: err instanceof Error ? err.message : String(err) });
    }
  });
};
