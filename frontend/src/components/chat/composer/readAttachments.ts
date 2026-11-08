// Reads picked files into attachments: images as data URLs, for a model
// with a vision projector, and everything else as text. An image for a
// model without one is skipped, since it could not be sent.

import type { Attachment } from "../state/chatStore.ts";

export function readAttachments(
  files: readonly File[],
  vision: boolean,
): Promise<Attachment[]> {
  const reads = files
    .filter((file) => vision || !isImage(file))
    .map((file) => readOne(file));
  return Promise.all(reads).then((all) =>
    all.filter((a): a is Attachment => a !== null),
  );
}

function isImage(file: File): boolean {
  return file.type.startsWith("image/");
}

function readOne(file: File): Promise<Attachment | null> {
  const type = isImage(file) ? "image" : "text";
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = () => {
      const content = reader.result;
      resolve(
        typeof content === "string"
          ? { name: file.name, size: file.size, type, content }
          : null,
      );
    };
    reader.onerror = () => resolve(null);
    if (type === "image") reader.readAsDataURL(file);
    else reader.readAsText(file);
  });
}
