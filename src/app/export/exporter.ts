import type { FolderStore, ImageEntry } from '../project/folder';
import { shapeBounds, type BoxShape, type ClassDef, type ImageDoc, type ProjectFile } from '../project/types';
import {
  safeFolderName,
  cocoJson,
  csvRows,
  dataYaml,
  exportable,
  readme,
  splitFor,
  vocXml,
  yoloLabel,
  SPLITS,
  type Format,
  type Split,
  type SplitSizes,
} from './formats';

export type Include = 'annotated' | 'done';
/** For classification: one image per annotation (cropped), or whole images sorted by their class. */
export type ClassifySource = 'crops' | 'images';

export interface ExportOptions {
  format: Format;
  /** Copy the image files into the export, so the dataset is self-contained. */
  includeImages: boolean;
  split: SplitSizes;
  include: Include;
  classifySource: ClassifySource;
}

export interface ExportPlan {
  docs: ImageDoc[];
  classes: ClassDef[];
  classIndex: Map<string, number>;
  annotations: number;
  unlabelled: number;
  /** Images left out: skipped, or (for 'done') not finished. */
  excluded: number;
}

export interface ExportSummary {
  images: number;
  annotations: number;
  filesWritten: number;
  /** Images whose EXIF says they should be rotated; their labels may not line up in other tools. */
  rotated: string[];
  /** Images left out of a classification export because they hold more than one class. */
  mixedClass: string[];
  counts: Record<Split, number>;
}

const baseName = (name: string) => name.replace(/\.[^.]+$/, '');
const clamp = (v: number, min: number, max: number) => Math.min(max, Math.max(min, v));

/** Read the EXIF orientation of a JPEG without decoding it. 1 means "no rotation". */
async function jpegOrientation(file: File) {
  if (!/\.jpe?g$/i.test(file.name)) return 1;
  try {
    const view = new DataView(await file.slice(0, 131072).arrayBuffer());
    if (view.byteLength < 8 || view.getUint16(0) !== 0xffd8) return 1;
    let offset = 2;
    while (offset + 4 < view.byteLength) {
      const marker = view.getUint16(offset);
      if ((marker & 0xff00) !== 0xff00) return 1;
      const size = view.getUint16(offset + 2);
      if (marker === 0xffe1 && view.getUint32(offset + 4) === 0x45786966) {
        const tiff = offset + 10;
        const little = view.getUint16(tiff) === 0x4949;
        const ifd = tiff + view.getUint32(tiff + 4, little);
        const entries = view.getUint16(ifd, little);
        for (let i = 0; i < entries; i++) {
          const entry = ifd + 2 + i * 12;
          if (entry + 12 > view.byteLength) break;
          if (view.getUint16(entry, little) === 0x0112) return view.getUint16(entry + 8, little);
        }
        return 1;
      }
      offset += 2 + size;
    }
  } catch {
    // A truncated or unusual header is not worth failing an export over.
  }
  return 1;
}

/** Work out what would be exported, reading the annotation files from disk. */
export async function planExport(folder: FolderStore, project: ProjectFile, images: ImageEntry[], options: ExportOptions): Promise<ExportPlan> {
  const classes = project.classes;
  const classIndex = new Map(classes.map((c, i) => [c.id, i]));
  const docs: ImageDoc[] = [];
  let annotations = 0;
  let unlabelled = 0;
  let excluded = 0;

  for (const image of images) {
    const summary = project.images[image.name];
    const status = summary?.status ?? 'todo';
    const wanted = options.include === 'done' ? status === 'done' : status !== 'skipped' && (summary?.annotations ?? 0) > 0;
    if (!wanted) {
      excluded++;
      continue;
    }
    const doc = await folder.readImageDoc(image.name);
    if (!doc) continue;
    docs.push(doc);
    annotations += exportable(doc, classIndex).length;
    unlabelled += doc.annotations.length - exportable(doc, classIndex).length;
  }
  return { docs, classes, classIndex, annotations, unlabelled, excluded };
}

/** Give a sync client or a virus scanner a moment to let go of the file before trying again. */
const ATTEMPTS = 3;
const RETRY_DELAY_MS = 250;
const wait = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Write one file, retrying a couple of times.
 *
 * The browser writes to a temporary file beside the target and swaps it in when the stream closes,
 * and it refuses the swap if anything else touched the file in the meantime. Dropbox, OneDrive and
 * virus scanners all do exactly that to files that have just appeared, which fails the write with
 * "the state had changed since it was read from disk". Waiting and writing again usually gets it.
 */
async function writeFile(dir: FileSystemDirectoryHandle, name: string, data: Blob | string) {
  let retried = false;
  for (let attempt = 1; ; attempt++) {
    let writable: FileSystemWritableFileStream | undefined;
    try {
      writable = await (await dir.getFileHandle(name, { create: true })).createWritable();
      await writable.write(data);
      await writable.close();
      // A write that failed part way leaves the browser's temporary file behind; it is not part of
      // the dataset, so don't leave it sitting in the export.
      if (retried) await dir.removeEntry(`${name}.crswap`).catch(() => undefined);
      return;
    } catch (err) {
      retried = true;
      // A stream that failed to close still holds the temporary file; let it go before retrying.
      await writable?.abort().catch(() => undefined);
      const detail = err instanceof Error ? err.message : String(err);
      if (err instanceof DOMException && err.name === 'NotAllowedError') throw new Error(`imagoLabel is no longer allowed to write to that folder (“${name}”).`);
      if (attempt >= ATTEMPTS) {
        throw new Error(
          `Couldn't write “${name}” after ${ATTEMPTS} tries: ${detail} ` +
            `This usually means something else on the computer is holding the file — if the folder you exported into is synced by Dropbox or OneDrive, ` +
            `pause syncing or export somewhere outside the synced folder, then try again.`,
        );
      }
      await wait(RETRY_DELAY_MS * attempt);
    }
  }
}

const writeText = (dir: FileSystemDirectoryHandle, name: string, text: string) => writeFile(dir, name, text);
const writeBlob = (dir: FileSystemDirectoryHandle, name: string, data: Blob) => writeFile(dir, name, data);

const subdir = (parent: FileSystemDirectoryHandle, ...names: string[]) =>
  names.reduce(async (dir, name) => (await dir).getDirectoryHandle(name, { create: true }), Promise.resolve(parent));

/** Cut one annotation's box out of the decoded image, as a JPEG. */
async function cropJpeg(bitmap: ImageBitmap, box: BoxShape) {
  const x = clamp(Math.round(box.x), 0, bitmap.width - 1);
  const y = clamp(Math.round(box.y), 0, bitmap.height - 1);
  const width = clamp(Math.round(box.width), 1, bitmap.width - x);
  const height = clamp(Math.round(box.height), 1, bitmap.height - y);
  const canvas = new OffscreenCanvas(width, height);
  canvas.getContext('2d')!.drawImage(bitmap, x, y, width, height, 0, 0, width, height);
  return canvas.convertToBlob({ type: 'image/jpeg', quality: 0.92 });
}

/** Write the dataset into `target`. Calls `onProgress` after each image. */
export async function runExport(
  target: FileSystemDirectoryHandle,
  folder: FolderStore,
  images: ImageEntry[],
  plan: ExportPlan,
  options: ExportOptions,
  onProgress: (done: number, total: number) => void,
): Promise<ExportSummary> {
  const { docs, classes, classIndex } = plan;
  const handles = new Map(images.map((i) => [i.name, i.handle]));
  const rotated: string[] = [];
  const mixedClass: string[] = [];
  let filesWritten = 0;

  const splits = new Map(docs.map((doc) => [doc.image.name, splitFor(doc.image.name, options.split)] as const));
  const counts: Record<Split, number> = { train: 0, val: 0, test: 0 };
  for (const split of splits.values()) counts[split]++;
  // A small dataset can produce an empty split even at 20%; don't point data.yaml at a folder that isn't there.
  const present = new Set(SPLITS.filter((s) => counts[s] > 0));
  const splitOf = (doc: ImageDoc) => splits.get(doc.image.name)!;

  const imageFile = async (doc: ImageDoc) => {
    const handle = handles.get(doc.image.name);
    if (!handle) return null;
    const file = await handle.getFile();
    if ((await jpegOrientation(file)) !== 1) rotated.push(doc.image.name);
    return file;
  };

  const copyImage = async (doc: ImageDoc, ...path: string[]) => {
    const file = await imageFile(doc);
    if (!file || !options.includeImages) return;
    await writeBlob(await subdir(target, ...path), file.name, file);
    filesWritten++;
  };

  if (options.format === 'yolo-classify') {
    for (const [i, doc] of docs.entries()) {
      const file = await imageFile(doc);
      const split = splitOf(doc);
      const annotations = exportable(doc, classIndex);
      if (file && annotations.length > 0) {
        if (options.classifySource === 'images') {
          const used = new Set(annotations.map((a) => a.classId));
          if (used.size === 1) {
            const dir = await subdir(target, split, safeFolderName(classes[classIndex.get(annotations[0].classId!)!].name));
            await writeBlob(dir, file.name, file);
            filesWritten++;
          } else {
            mixedClass.push(doc.image.name);
          }
        } else {
          // One cropped image per annotation, so several classes in one photo all get used.
          const bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' });
          for (const [index, annotation] of annotations.entries()) {
            const dir = await subdir(target, split, safeFolderName(classes[classIndex.get(annotation.classId!)!].name));
            await writeBlob(dir, `${baseName(doc.image.name)}_${index + 1}.jpg`, await cropJpeg(bitmap, shapeBounds(annotation.shape)));
            filesWritten++;
          }
          bitmap.close();
        }
      }
      onProgress(i + 1, docs.length);
    }
  } else if (options.format.startsWith('yolo')) {
    for (const [i, doc] of docs.entries()) {
      const split = splitOf(doc);
      await writeText(await subdir(target, 'labels', split), `${baseName(doc.image.name)}.txt`, yoloLabel(doc, classIndex, options.format));
      filesWritten++;
      await copyImage(doc, 'images', split);
      onProgress(i + 1, docs.length);
    }
    await writeText(target, 'data.yaml', dataYaml(classes, present));
    filesWritten++;
  } else if (options.format === 'coco') {
    for (const [i, doc] of docs.entries()) {
      await copyImage(doc, 'images', splitOf(doc));
      onProgress(i + 1, docs.length);
    }
    if (present.size > 1) {
      for (const split of SPLITS.filter((s) => present.has(s))) {
        await writeText(target, `annotations_${split}.json`, cocoJson({ docs: docs.filter((d) => splitOf(d) === split), classes, classIndex }));
        filesWritten++;
      }
    } else {
      await writeText(target, 'annotations.json', cocoJson({ docs, classes, classIndex }));
      filesWritten++;
    }
  } else if (options.format === 'voc') {
    for (const [i, doc] of docs.entries()) {
      await writeText(await subdir(target, 'Annotations'), `${baseName(doc.image.name)}.xml`, vocXml(doc, classes, classIndex, folder.name));
      filesWritten++;
      await copyImage(doc, 'JPEGImages');
      onProgress(i + 1, docs.length);
    }
    const sets = await subdir(target, 'ImageSets', 'Main');
    for (const split of SPLITS.filter((s) => present.has(s))) {
      const names = docs.filter((d) => splitOf(d) === split).map((d) => baseName(d.image.name));
      await writeText(sets, `${split}.txt`, names.join('\n') + '\n');
      filesWritten++;
    }
  } else {
    for (const [i, doc] of docs.entries()) {
      await copyImage(doc, 'images');
      onProgress(i + 1, docs.length);
    }
    await writeText(target, 'annotations.csv', csvRows(docs, classes, classIndex, present.size > 1 ? splits : undefined));
    filesWritten++;
  }

  await writeText(target, 'README.txt', readme(options.format, options.includeImages || options.format === 'yolo-classify'));
  filesWritten++;
  return { images: docs.length, annotations: plan.annotations, filesWritten, rotated, mixedClass, counts };
}
