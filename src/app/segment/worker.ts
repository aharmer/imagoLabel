// Runs SAM 2.1 off the main thread. The heavy image encoder runs once per image (on the GPU when
// possible); each click only runs the small mask decoder, which always runs on the CPU.
import { AutoModel, AutoProcessor, RawImage, Tensor } from '@huggingface/transformers';
import { addMask, candidateMasks, fillHoles, keepBestRegion, simplifyRing, subtractMask, traceOuterContour, upsampleMask } from '../../shared/mask';
import { configureOnnxRuntime } from '../../shared/ort';
import type { Device, FromSegmentWorker, PromptPoint, Region, SegmentResult, ToSegmentWorker } from './protocol';

interface WorkerScope {
  postMessage(message: FromSegmentWorker): void;
  onmessage: ((event: MessageEvent<ToSegmentWorker>) => void) | null;
}
const scope = self as unknown as WorkerScope;
const post = (message: FromSegmentWorker) => scope.postMessage(message);

configureOnnxRuntime();

const MODEL = {
  repo: 'onnx-community/sam2.1-hiera-small-ONNX',
  gpu: { vision_encoder: 'fp16', prompt_encoder_mask_decoder: 'fp32' },
  cpu: { vision_encoder: 'q8', prompt_encoder_mask_decoder: 'q8' },
} as const;
/** Encoded images kept in memory (~16 MB each), so returning to a recent image is instant. */
const MAX_ENCODED = 6;
/** Polygon simplification tolerance, in pixels of the encoded picture (≤1024 px on its long side). */
const SIMPLIFY_TOLERANCE = 1;
/** A reading covering this much of the picture is the background, not the thing that was clicked. */
const MOST_OF_THE_PICTURE = 0.85;

// Transformers.js model and processor types are too loose to be useful here.
let model: any = null;
let processor: any = null;
let device: Device = 'wasm';

interface Encoded {
  embeddings: Record<string, Tensor>;
  inputs: any;
  region: Region;
  width: number;
  height: number;
}
const encoded = new Map<string, Encoded>();

/**
 * The object being outlined: which prompts produced it, and the mask behind each of the options
 * the user was offered. A request whose prompts extend these is another click on the same object,
 * so it starts from the option they are looking at.
 */
let current: { key: string; prompts: string[]; masks: Uint8Array[] } | null = null;

const promptSignature = (points: PromptPoint[], box: Region | null) => [
  ...(box ? [`box ${box.x},${box.y},${box.width},${box.height}`] : []),
  ...points.map((p) => `${p.x},${p.y}${p.positive ? '+' : '-'}`),
];

const isPrefix = (a: string[], b: string[]) => a.every((v, i) => v === b[i]);

async function load(target: Device) {
  const files = new Map<string, { loaded: number; total: number }>();
  let lastPost = 0;
  const progress_callback = (p: any) => {
    if (p.status !== 'progress' || !p.file) return;
    files.set(p.file, { loaded: p.loaded ?? 0, total: p.total ?? 0 });
    if (performance.now() - lastPost < 150) return;
    lastPost = performance.now();
    let loaded = 0;
    let total = 0;
    for (const f of files.values()) {
      loaded += f.loaded;
      total += f.total;
    }
    post({ type: 'progress', loaded, total });
  };
  await model?.dispose?.();
  encoded.clear();
  current = null;
  processor ??= await AutoProcessor.from_pretrained(MODEL.repo, { progress_callback });
  model = await AutoModel.from_pretrained(MODEL.repo, {
    device: { vision_encoder: target, prompt_encoder_mask_decoder: 'wasm' },
    dtype: target === 'webgpu' ? MODEL.gpu : MODEL.cpu,
    progress_callback,
  });
  device = target;
}

async function gpuUsable() {
  try {
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
    return Boolean(adapter?.features.has('shader-f16'));
  } catch {
    return false;
  }
}

async function init(preferGpu: boolean) {
  if (preferGpu && (await gpuUsable())) {
    try {
      await load('webgpu');
      return;
    } catch (err) {
      console.warn('WebGPU model failed to load; falling back to CPU.', err);
    }
  }
  await load('wasm');
}

async function encode(key: string, bitmap: ImageBitmap, region: Region) {
  const { width, height } = bitmap;
  const ctx = new OffscreenCanvas(width, height).getContext('2d')!;
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();
  const raw = new RawImage(ctx.getImageData(0, 0, width, height).data, width, height, 4).rgb();
  const inputs = await processor(raw);
  let embeddings: Record<string, Tensor>;
  try {
    embeddings = await model.get_image_embeddings(inputs);
  } catch (err) {
    if (device !== 'webgpu') throw err;
    // Some GPUs/drivers fail at run time rather than load time; the CPU always works.
    console.warn('WebGPU encoding failed; switching to CPU.', err);
    await load('wasm');
    embeddings = await model.get_image_embeddings(inputs);
  }
  encoded.delete(key);
  encoded.set(key, { embeddings, inputs, region, width, height });
  while (encoded.size > MAX_ENCODED) encoded.delete(encoded.keys().next().value!);
}

async function segment(key: string, points: PromptPoint[], box: Region | null, level: number): Promise<SegmentResult> {
  const entry = encoded.get(key)!;
  // Mark as recently used.
  encoded.delete(key);
  encoded.set(key, entry);
  const { region, width, height, inputs } = entry;
  const sx = width / region.width;
  const sy = height / region.height;
  const toLocal = (x: number, y: number): [number, number] => [(x - region.x) * sx, (y - region.y) * sy];

  // Clicks in the encoded picture's own pixels, which is what the masks are measured against.
  const local = points.map((p) => {
    const [x, y] = toLocal(p.x, p.y);
    return { x, y, positive: p.positive };
  });
  const prompts = promptSignature(points, box);
  // Another click on the same object builds on what the user can see; stepping back with Backspace
  // shortens the prompts instead, and then there is nothing to build on.
  const adding = current?.key === key && isPrefix(current.prompts, prompts) && prompts.length > current.prompts.length;
  const previous = adding ? current!.masks[Math.min(level, current!.masks.length - 1)] : null;
  const last = points[points.length - 1];
  const removing = Boolean(previous && last && !last.positive);

  // Asking SAM to exclude a point makes it abandon the whole object and offer slivers around the
  // point instead, so a right-click is not sent as a negative prompt. We ask what is at that spot
  // and take that away, which is also what "remove this bit" means to the person clicking.
  const asked = removing ? [last] : points.filter((p) => p.positive);
  const prompt: Record<string, Tensor> = {};
  if (asked.length) {
    prompt.input_points = processor.reshape_input_points([[asked.map((p) => toLocal(p.x, p.y))]], inputs.original_sizes, inputs.reshaped_input_sizes);
    prompt.input_labels = new Tensor('int64', BigInt64Array.from(asked.map(() => 1n)), [1, 1, asked.length]);
  }
  if (box && !removing) {
    const [x1, y1] = toLocal(box.x, box.y);
    const [x2, y2] = toLocal(box.x + box.width, box.y + box.height);
    prompt.input_boxes = processor.reshape_input_points([[[x1, y1, x2, y2]]], inputs.original_sizes, inputs.reshaped_input_sizes, true);
  }
  const outputs = await model({ ...entry.embeddings, ...prompt });

  const ip = processor.image_processor;
  const candidates = candidateMasks(outputs, inputs.reshaped_input_sizes[0], ip.do_pad && ip.pad_size ? ip.pad_size : null);
  const seeds = local.filter((p) => p.positive).map((p): [number, number] => [p.x, p.y]);
  if (!seeds.length && box) seeds.push(toLocal(box.x + box.width / 2, box.y + box.height / 2));

  const readings = candidates.map((candidate) => {
    const reading = upsampleMask(candidate, width, height);
    // A click only ever adds to the outline and a right-click only ever takes away, so extending an
    // object can never lose the part of it that is already outlined.
    const mask = previous ? previous.slice() : reading;
    if (previous && removing) subtractMask(mask, reading);
    else if (previous) addMask(mask, reading);

    keepBestRegion(mask, width, height, seeds);
    fillHoles(mask, width, height);
    let area = 0;
    for (let i = 0; i < mask.length; i++) area += mask[i];
    // Does the outline this would leave match every click the user has made?
    let agrees = 0;
    for (const p of local) if ((mask[Math.round(p.y) * width + Math.round(p.x)] === 1) === p.positive) agrees++;
    return { mask, area, agrees, score: candidate.score };
  });

  // Biggest outline first, so a right-click's options run from the smallest bite taken out to the
  // largest. SAM rates a small part of an object above the whole of it often enough that
  // going by its score alone makes one click select a fragment, so prefer the reading that agrees
  // with every click and then simply the most generous one — short of swallowing the whole picture.
  readings.sort((a, b) => b.area - a.area);
  const roomy = width * height * MOST_OF_THE_PICTURE;
  const rank = (r: (typeof readings)[number]): [number, number, number] => [r.agrees, r.area <= roomy ? 1 : 0, r.area];
  let chosen = 0;
  for (let i = 1; i < readings.length; i++) {
    const [agrees, fits, area] = rank(readings[i]);
    const [bestAgrees, bestFits, bestArea] = rank(readings[chosen]);
    if (agrees > bestAgrees || (agrees === bestAgrees && (fits > bestFits || (fits === bestFits && area > bestArea)))) chosen = i;
  }

  // A reading too small to trace is no use as an option, so drop it and keep the rest aligned with
  // the masks the next click will build on.
  const traced = readings.map((r) => ({ ...r, ring: simplifyRing(traceOuterContour(r.mask, width, height), SIMPLIFY_TOLERANCE) }));
  const usable = traced.filter((r) => r.ring.length >= 3);
  if (!usable.length) {
    current = null;
    return { options: [], chosen: 0 };
  }
  current = { key, prompts, masks: usable.map((r) => r.mask) };

  const round = (v: number) => Math.round(v * 100) / 100;
  return {
    options: usable.map((r) => ({
      polygon: r.ring.map(([x, y]): [number, number] => [round(region.x + (x + 0.5) / sx), round(region.y + (y + 0.5) / sy)]),
      area: r.area,
      score: r.score,
    })),
    chosen: Math.max(0, usable.indexOf(traced[chosen])),
  };
}

// Handle one message at a time; the page decides the order (clicks before background work).
let queue = Promise.resolve();
scope.onmessage = (event) => {
  const msg = event.data;
  queue = queue.then(async () => {
    try {
      if (msg.type === 'init') {
        await init(msg.preferGpu);
        post({ type: 'ready', id: msg.id, device });
      } else if (msg.type === 'encode') {
        await encode(msg.key, msg.bitmap, msg.region);
        post({ type: 'encoded', id: msg.id });
      } else if (!encoded.has(msg.key)) {
        post({ type: 'error', id: msg.id, message: 'Image is not encoded', notEncoded: true });
      } else {
        post({ type: 'segmented', id: msg.id, result: await segment(msg.key, msg.points, msg.box, msg.level) });
      }
    } catch (err) {
      post({ type: 'error', id: msg.id, message: err instanceof Error ? err.message : String(err) });
    }
  });
};
