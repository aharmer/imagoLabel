import { create } from 'zustand';
import type { Device, FromSegmentWorker, PromptPoint, Region, SegmentResult, ToSegmentWorker } from './protocol';

export type ModelStatus =
  | { kind: 'idle' }
  | { kind: 'loading'; loaded: number; total: number }
  | { kind: 'ready'; device: Device }
  | { kind: 'error'; message: string };

export const useModelStatus = create<{ status: ModelStatus }>(() => ({ status: { kind: 'idle' } }));
const setStatus = (status: ModelStatus) => useModelStatus.setState({ status });

/** Keep the page's picture of what's encoded in step with the worker's cache size. */
const MAX_ENCODED = 6;
/** Longest side of the picture handed to the encoder; SAM works at 1024 px. */
export const ENCODE_SIZE = 1024;

type Priority = 'user' | 'background';
/** A worker message before the queue assigns its id. */
type Message = ToSegmentWorker extends infer T ? (T extends unknown ? Omit<T, 'id'> : never) : never;

interface Task {
  priority: Priority;
  message: Message;
  transfer: Transferable[];
  resolve: (value: FromSegmentWorker) => void;
  reject: (reason: Error) => void;
}

class NotEncodedError extends Error {}

/**
 * Talks to the segmentation worker. Sends one request at a time so a click never waits behind
 * more than one piece of background work, and user requests jump ahead of queued background ones.
 */
class Segmenter {
  private worker: Worker | null = null;
  private ready: Promise<void> | null = null;
  private queue: Task[] = [];
  private inFlight: { id: number; task: Task } | null = null;
  private nextId = 1;
  /** Keys that are encoded (or being encoded), most recently used last. */
  private encoded = new Map<string, Promise<void>>();

  /** Start loading the model (downloads it the first time). Safe to call repeatedly. */
  load(): Promise<void> {
    if (this.ready) return this.ready;
    this.worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
    this.worker.onmessage = (event: MessageEvent<FromSegmentWorker>) => this.onMessage(event.data);
    this.worker.onerror = (event) => {
      event.preventDefault();
      this.fail(new Error(event.message || 'The segmentation model crashed, most likely by running out of memory.'));
    };
    setStatus({ kind: 'loading', loaded: 0, total: 0 });
    this.ready = this.send('user', { type: 'init', preferGpu: true }).then(
      (msg) => {
        if (msg.type === 'ready') setStatus({ kind: 'ready', device: msg.device });
      },
      (err: Error) => {
        setStatus({ kind: 'error', message: err.message });
        this.reset();
        throw err;
      },
    );
    return this.ready;
  }

  isEncoded(key: string) {
    return this.encoded.has(key);
  }

  /** Encode a region of an image under `key`, unless it already is. `picture` is only called when needed. */
  encode(key: string, region: Region, picture: () => Promise<ImageBitmap>, priority: Priority): Promise<void> {
    const existing = this.encoded.get(key);
    if (existing) {
      this.touch(key, existing);
      if (priority === 'user') this.promote(key);
      return existing;
    }
    const pending = this.load()
      .then(picture)
      .then((bitmap) => this.send(priority, { type: 'encode', key, bitmap, region }, [bitmap]))
      .then(() => undefined);
    pending.catch(() => this.encoded.delete(key));
    this.touch(key, pending);
    return pending;
  }

  /** Segment with point and/or box prompts, encoding first if needed. */
  async segment(key: string, region: Region, picture: () => Promise<ImageBitmap>, points: PromptPoint[], box: Region | null, level: number): Promise<SegmentResult> {
    for (let attempt = 0; ; attempt++) {
      await this.encode(key, region, picture, 'user');
      try {
        const msg = await this.send('user', { type: 'segment', key, points, box, level });
        if (msg.type === 'segmented') return msg.result;
        throw new Error('Unexpected response from segmentation worker');
      } catch (err) {
        // The worker evicted this image from its cache; encode it again once.
        if (err instanceof NotEncodedError && attempt === 0) {
          this.encoded.delete(key);
          continue;
        }
        throw err;
      }
    }
  }

  /** Drop queued background work, e.g. pre-encoding for an image the user has moved away from. */
  cancelBackground() {
    const dropped = this.queue.filter((t) => t.priority === 'background');
    this.queue = this.queue.filter((t) => t.priority !== 'background');
    for (const task of dropped) task.reject(new Error('Cancelled'));
  }

  private touch(key: string, pending: Promise<void>) {
    this.encoded.delete(key);
    this.encoded.set(key, pending);
    while (this.encoded.size > MAX_ENCODED) this.encoded.delete(this.encoded.keys().next().value!);
  }

  /** A queued background encode that the user now needs: move it to the front. */
  private promote(key: string) {
    const index = this.queue.findIndex((t) => t.message.type === 'encode' && t.message.key === key);
    if (index > 0) {
      const [task] = this.queue.splice(index, 1);
      task.priority = 'user';
      this.queue.unshift(task);
    }
  }

  private send(priority: Priority, message: Message, transfer: Transferable[] = []) {
    return new Promise<FromSegmentWorker>((resolve, reject) => {
      const task: Task = { priority, message, transfer, resolve, reject };
      if (priority === 'user') {
        const firstBackground = this.queue.findIndex((t) => t.priority === 'background');
        this.queue.splice(firstBackground < 0 ? this.queue.length : firstBackground, 0, task);
      } else {
        this.queue.push(task);
      }
      this.pump();
    });
  }

  private pump() {
    if (this.inFlight || !this.worker) return;
    const task = this.queue.shift();
    if (!task) return;
    const id = this.nextId++;
    this.inFlight = { id, task };
    this.worker.postMessage({ ...task.message, id } as ToSegmentWorker, task.transfer);
  }

  private onMessage(msg: FromSegmentWorker) {
    if (msg.type === 'progress') {
      setStatus({ kind: 'loading', loaded: msg.loaded, total: msg.total });
      return;
    }
    const current = this.inFlight;
    if (!current || current.id !== msg.id) return;
    this.inFlight = null;
    if (msg.type === 'error') current.task.reject(msg.notEncoded ? new NotEncodedError(msg.message) : new Error(msg.message));
    else current.task.resolve(msg);
    if (msg.type === 'ready') setStatus({ kind: 'ready', device: msg.device });
    this.pump();
  }

  private fail(err: Error) {
    this.inFlight?.task.reject(err);
    for (const task of this.queue) task.reject(err);
    this.queue = [];
    this.inFlight = null;
    setStatus({ kind: 'error', message: err.message });
    this.reset();
  }

  /** Forget the worker so the next load() starts a fresh one. */
  private reset() {
    this.worker?.terminate();
    this.worker = null;
    this.ready = null;
    this.encoded.clear();
  }
}

export const segmenter = new Segmenter();

/** Cut `region` out of a displayed image and scale it so its long side is ENCODE_SIZE. */
export function regionPicture(bitmap: ImageBitmap, imageWidth: number, region: Region): Promise<ImageBitmap> {
  const k = bitmap.width / imageWidth;
  const scale = ENCODE_SIZE / Math.max(region.width, region.height);
  return createImageBitmap(bitmap, region.x * k, region.y * k, region.width * k, region.height * k, {
    resizeWidth: Math.max(1, Math.round(region.width * scale)),
    resizeHeight: Math.max(1, Math.round(region.height * scale)),
    resizeQuality: 'high',
  });
}
