'use strict';
/**
 * The complete inventory of what the page may do natively.
 *
 * There is no "read this file" and no "run this command" here. The renderer can ask for
 * one of two named background stages, read version information, and open an external
 * https link. Everything else it needs comes over HTTP from the local server. This
 * application renders data fetched from public block explorers, so the page is treated as
 * handling untrusted input throughout.
 */

const { contextBridge, ipcRenderer } = require('electron');

const CHANNELS = ['stage:start', 'stage:line', 'stage:done', 'nav'];

contextBridge.exposeInMainWorld('desktop', {
  isDesktop: true,

  ingest: (opts) => ipcRenderer.invoke('stage:ingest', opts || {}),
  analyse: () => ipcRenderer.invoke('stage:analyse'),
  info: () => ipcRenderer.invoke('app:info'),
  openExternal: (url) => ipcRenderer.invoke('shell:openExternal', url),

  on: (channel, handler) => {
    if (!CHANNELS.includes(channel)) return () => {};
    // Pass only the payload through; the event object carries a sender reference.
    const wrapped = (event, payload) => handler(payload);
    ipcRenderer.on(channel, wrapped);
    return () => ipcRenderer.removeListener(channel, wrapped);
  },
});
