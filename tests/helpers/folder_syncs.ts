/**
 * Observe the folder syncs `syncFolder` (`src/platform.ts`) makes, by spying on `fs`.
 */
import { spyOn } from "bun:test";
import fs from "node:fs";

/**
 * Run `fn` and return the real paths of the folders synced while it ran, in order. With `fail`,
 * each folder's fsync throws a system error with that code instead; files sync as usual.
 */
export function folderSyncs(fn: () => unknown, fail?: string): string[] {
  const synced: string[] = [];
  const folders = new Map<number, string>();
  const { openSync, fsyncSync } = fs;
  const spies = [
    spyOn(fs, "openSync").mockImplementation(((target: fs.PathLike, ...rest: unknown[]) => {
      const fd = (openSync as (...args: unknown[]) => number)(target, ...rest);
      // A descriptor number is reused once closed, so each open replaces what it held.
      if (fs.fstatSync(fd).isDirectory()) folders.set(fd, fs.realpathSync(String(target)));
      else folders.delete(fd);
      return fd;
    }) as typeof fs.openSync),
    spyOn(fs, "fsyncSync").mockImplementation((fd) => {
      const folder = folders.get(fd);
      if (folder !== undefined && fail !== undefined) {
        throw Object.assign(new Error(`${fail}: fsync failed`), { code: fail, syscall: "fsync", errno: -5 });
      }
      fsyncSync(fd);
      if (folder !== undefined) synced.push(folder);
    }),
  ];
  try {
    fn();
  } finally {
    for (const spy of spies) spy.mockRestore();
  }
  return synced;
}
