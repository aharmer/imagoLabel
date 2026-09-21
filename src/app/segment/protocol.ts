export type Device = 'webgpu' | 'wasm';

/** A rectangle in original image pixels. */
export interface Region {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface PromptPoint {
  x: number;
  y: number;
  /** true = part of the object, false = not part of it. */
  positive: boolean;
}

export type ToSegmentWorker =
  | { type: 'init'; id: number; preferGpu: boolean }
  /** Encode a (downscaled) picture of `region`; later prompts on `key` reuse the result. */
  | { type: 'encode'; id: number; key: string; bitmap: ImageBitmap; region: Region }
  /** `level` is the option the user is looking at, so the next click builds on what they can see. */
  | { type: 'segment'; id: number; key: string; points: PromptPoint[]; box: Region | null; level: number };

/** One reading of the clicks so far: the whole object, or a smaller part of it. */
export interface SegmentOption {
  /** Closed ring in original image pixels. */
  polygon: Array<[number, number]>;
  /** How much it covers, in the encoded picture's pixels. */
  area: number;
  score: number;
}

export type SegmentResult = {
  /** Biggest first, so stepping through them is "less of the object" each time. */
  options: SegmentOption[];
  /** Which one to show. */
  chosen: number;
};

export type FromSegmentWorker =
  | { type: 'progress'; loaded: number; total: number }
  | { type: 'ready'; id: number; device: Device }
  | { type: 'encoded'; id: number }
  | { type: 'segmented'; id: number; result: SegmentResult }
  | { type: 'error'; id: number; message: string; notEncoded?: boolean };
