// Turning imagoLabel's annotations into the file formats training pipelines expect.
// Every format takes coordinates in original image pixels and writes whatever units it needs.
import { shapeBounds, type Annotation, type ClassDef, type ImageDoc } from '../project/types';

export type Format = 'yolo-detect' | 'yolo-segment' | 'yolo-classify' | 'coco' | 'voc' | 'csv';

export type Split = 'train' | 'val' | 'test';
export const SPLITS: Split[] = ['train', 'val', 'test'];

/**
 * What the split folders are called in a detection or segmentation export. data.yaml names them
 * outright, so these follow the spelling Roboflow uses and people expect to see.
 *
 * Classification exports keep "val": they have no yaml, so Ultralytics goes looking for the folder
 * by name, and only recent versions accept "valid" — an older one would train with no validation
 * set at all and not say so.
 */
export const SPLIT_DIR: Record<Split, string> = { train: 'train', val: 'valid', test: 'test' };

/** Percentages held back from training; the rest is training data. */
export interface SplitSizes {
  val: number;
  test: number;
}

export const FORMAT_LABEL: Record<Format, string> = {
  'yolo-detect': 'YOLO — boxes (Ultralytics detect)',
  'yolo-segment': 'YOLO — polygons (Ultralytics segment)',
  'yolo-classify': 'YOLO — classification (Ultralytics classify)',
  coco: 'COCO JSON',
  voc: 'Pascal VOC XML',
  csv: 'CSV',
};

export const FORMAT_NOTE: Record<Format, string> = {
  'yolo-detect': 'train/valid/test, each holding images and labels, plus data.yaml. One box per line. Polygons are converted to their bounding box.',
  'yolo-segment': 'train/valid/test, each holding images and labels, plus data.yaml. One polygon per line. Boxes are written as four-corner polygons.',
  'yolo-classify': 'Folders of images per class (train/<class>/image.jpg). Each annotation is cropped out, or whole images are sorted by class.',
  coco: 'A single annotations.json holding boxes and polygons.',
  voc: 'One .xml per image, boxes only. Polygons are converted to their bounding box.',
  csv: 'One row per annotation: class, box, and the polygon when there is one.',
};

/** Formats that describe a polygon; the others get bounding boxes. */
export const keepsPolygons = (format: Format) => format === 'yolo-segment' || format === 'coco' || format === 'csv';

const clamp01 = (v: number) => Math.min(1, Math.max(0, v));
const n = (v: number, decimals = 6) => Number(v.toFixed(decimals)).toString();

/** Annotations that can be exported: those with a class. */
export const exportable = (doc: ImageDoc, classIndex: Map<string, number>) =>
  doc.annotations.filter((a) => a.classId !== null && classIndex.has(a.classId));

function polygonPoints(annotation: Annotation): Array<[number, number]> {
  if (annotation.shape.type === 'polygon') return annotation.shape.points;
  const { x, y, width, height } = annotation.shape;
  return [
    [x, y],
    [x + width, y],
    [x + width, y + height],
    [x, y + height],
  ];
}

export function yoloLabel(doc: ImageDoc, classIndex: Map<string, number>, format: Format) {
  const { width, height } = doc.image;
  const lines = exportable(doc, classIndex).map((a) => {
    const index = classIndex.get(a.classId!)!;
    if (format === 'yolo-segment') {
      const coords = polygonPoints(a).flatMap(([x, y]) => [n(clamp01(x / width)), n(clamp01(y / height))]);
      return `${index} ${coords.join(' ')}`;
    }
    const box = shapeBounds(a.shape);
    return `${index} ${n(clamp01((box.x + box.width / 2) / width))} ${n(clamp01((box.y + box.height / 2) / height))} ${n(clamp01(box.width / width))} ${n(clamp01(box.height / height))}`;
  });
  return lines.join('\n') + (lines.length ? '\n' : '');
}

/** Ultralytics resolves relative train/val paths against the folder holding data.yaml when `path` is omitted. */
/** Only detection and segmentation use a data.yaml; classification takes its classes from folder names. */
export function dataYaml(classes: ClassDef[], present: Set<Split>) {
  const names = classes.map((c, i) => `  ${i}: ${JSON.stringify(c.name)}`).join('\n');
  const dir = (split: Split) => `${SPLIT_DIR[split]}/images`;
  const lines = [
    '# Dataset exported by imagoLabel (https://imago-label.vercel.app)',
    `# Created ${new Date().toISOString()}`,
    '',
    `train: ${dir('train')}`,
    `val: ${dir(present.has('val') ? 'val' : 'train')}`,
  ];
  if (present.has('test')) lines.push(`test: ${dir('test')}`);
  return [...lines, '', 'names:', names, ''].join('\n');
}

export function readme(format: Format, hasImages: boolean) {
  const lines = [
    'Dataset exported by imagoLabel',
    '==============================',
    '',
    `Format: ${FORMAT_LABEL[format]}`,
    `Created: ${new Date().toISOString()}`,
    '',
  ];
  if (format === 'yolo-classify') {
    lines.push(
      'Train with Ultralytics:',
      '',
      '  yolo classify train data=. model=yolo11n-cls.pt epochs=100 imgsz=224',
      '',
      'Each split folder holds one sub-folder per class. Ultralytics takes the class names from the',
      'folder names, so this layout needs no data.yaml. The validation folder is "val" rather than',
      '"valid" because Ultralytics finds it by name, and older versions only recognise "val".',
    );
  } else if (format.startsWith('yolo')) {
    lines.push(
      'Train with Ultralytics:',
      '',
      `  yolo ${format === 'yolo-segment' ? 'segment' : 'detect'} train data=data.yaml model=${format === 'yolo-segment' ? 'yolo11n-seg.pt' : 'yolo11n.pt'} epochs=100 imgsz=640`,
      '',
      'Layout:',
      '',
      '  data.yaml',
      '  train/images/   train/labels/',
      '  valid/images/   valid/labels/',
      '  test/images/    test/labels/   (only when a test split was asked for)',
      '',
      'Label files use normalised coordinates (0-1) and class indices matching the order in data.yaml.',
    );
    if (!hasImages) {
      lines.push(
        '',
        'This export contains labels only. Copy your images into train/images and valid/images',
        'so each image sits beside the label file of the same name.',
      );
    }
  } else if (format === 'coco') {
    lines.push('annotations.json follows the COCO instance format: bbox is [x, y, width, height] in pixels,', 'and segmentation holds polygons as [x1, y1, x2, y2, ...].');
  } else if (format === 'voc') {
    lines.push('One XML file per image in Annotations/, in the Pascal VOC layout.');
  } else {
    lines.push('annotations.csv has one row per annotation. Boxes are in pixels (xmin, ymin, xmax, ymax).');
  }
  return lines.join('\n') + '\n';
}

export interface CocoOptions {
  docs: ImageDoc[];
  classes: ClassDef[];
  classIndex: Map<string, number>;
}

export function cocoJson({ docs, classes, classIndex }: CocoOptions) {
  const images = docs.map((doc, i) => ({ id: i + 1, file_name: doc.image.name, width: doc.image.width, height: doc.image.height }));
  const annotations: unknown[] = [];
  docs.forEach((doc, i) => {
    for (const a of exportable(doc, classIndex)) {
      const box = shapeBounds(a.shape);
      annotations.push({
        id: annotations.length + 1,
        image_id: i + 1,
        category_id: classIndex.get(a.classId!)! + 1,
        bbox: [box.x, box.y, box.width, box.height].map((v) => Number(v.toFixed(2))),
        area: Number((box.width * box.height).toFixed(2)),
        iscrowd: 0,
        segmentation: a.shape.type === 'polygon' ? [a.shape.points.flat().map((v) => Number(v.toFixed(2)))] : [],
      });
    }
  });
  return JSON.stringify(
    {
      info: { description: 'Exported by imagoLabel', date_created: new Date().toISOString() },
      images,
      annotations,
      categories: classes.map((c, i) => ({ id: i + 1, name: c.name, supercategory: 'none' })),
    },
    null,
    1,
  );
}

const xmlEscape = (value: string) => value.replace(/[<>&'"]/g, (c) => ({ '<': '&lt;', '>': '&gt;', '&': '&amp;', "'": '&apos;', '"': '&quot;' })[c]!);

export function vocXml(doc: ImageDoc, classes: ClassDef[], classIndex: Map<string, number>, folderName: string) {
  const objects = exportable(doc, classIndex).map((a) => {
    const box = shapeBounds(a.shape);
    return [
      '  <object>',
      `    <name>${xmlEscape(classes[classIndex.get(a.classId!)!].name)}</name>`,
      '    <pose>Unspecified</pose>',
      '    <truncated>0</truncated>',
      '    <difficult>0</difficult>',
      '    <bndbox>',
      `      <xmin>${Math.round(box.x)}</xmin>`,
      `      <ymin>${Math.round(box.y)}</ymin>`,
      `      <xmax>${Math.round(box.x + box.width)}</xmax>`,
      `      <ymax>${Math.round(box.y + box.height)}</ymax>`,
      '    </bndbox>',
      '  </object>',
    ].join('\n');
  });
  return [
    '<annotation>',
    `  <folder>${xmlEscape(folderName)}</folder>`,
    `  <filename>${xmlEscape(doc.image.name)}</filename>`,
    '  <source><database>imagoLabel</database></source>',
    '  <size>',
    `    <width>${doc.image.width}</width>`,
    `    <height>${doc.image.height}</height>`,
    '    <depth>3</depth>',
    '  </size>',
    '  <segmented>0</segmented>',
    ...objects,
    '</annotation>',
    '',
  ].join('\n');
}

const csvCell = (value: string | number) => {
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
};

export function csvRows(docs: ImageDoc[], classes: ClassDef[], classIndex: Map<string, number>, splits?: Map<string, string>) {
  const header = ['image', 'image_width', 'image_height', 'class', 'xmin', 'ymin', 'xmax', 'ymax', 'shape', 'polygon', 'source', 'score'];
  const rows = [(splits ? ['split', ...header] : header).join(',')];
  for (const doc of docs) {
    for (const a of exportable(doc, classIndex)) {
      const box = shapeBounds(a.shape);
      rows.push(
        [
          ...(splits ? [splits.get(doc.image.name) ?? 'train'] : []),
          csvCell(doc.image.name),
          doc.image.width,
          doc.image.height,
          csvCell(classes[classIndex.get(a.classId!)!].name),
          Math.round(box.x),
          Math.round(box.y),
          Math.round(box.x + box.width),
          Math.round(box.y + box.height),
          a.shape.type,
          csvCell(a.shape.type === 'polygon' ? a.shape.points.map(([x, y]) => `${Math.round(x)} ${Math.round(y)}`).join(' ') : ''),
          a.source,
          a.score === undefined ? '' : a.score.toFixed(3),
        ].join(','),
      );
    }
  }
  return rows.join('\n') + '\n';
}

/**
 * Deterministic split: the same image always lands in the same set, so re-exporting after
 * annotating more images doesn't shuffle images between training, validation and test.
 */
export function splitFor(name: string, sizes: SplitSizes): Split {
  let hash = 2166136261;
  for (let i = 0; i < name.length; i++) {
    hash ^= name.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  const bucket = (hash >>> 0) % 100;
  if (bucket < sizes.val) return 'val';
  if (bucket < sizes.val + sizes.test) return 'test';
  return 'train';
}

/** Turn a class or dataset name into something safe to use as a folder name. */
export const safeFolderName = (name: string) => name.trim().replace(/[<>:"/\\|?*\u0000-\u001f]/g, '_').replace(/[. ]+$/, '') || 'unnamed';

const FORMAT_SUFFIX: Record<Format, string> = {
  'yolo-detect': 'yolo',
  'yolo-segment': 'yolo-seg',
  'yolo-classify': 'yolo-cls',
  coco: 'coco',
  voc: 'voc',
  csv: 'csv',
};

/** Suggested name for the dataset folder, e.g. "beetles-yolo". */
export const suggestedFolderName = (imageFolder: string, format: Format) => safeFolderName(`${imageFolder}-${FORMAT_SUFFIX[format]}`);
