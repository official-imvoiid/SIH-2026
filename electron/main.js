'use strict';
/**
 * Desktop shell.
 *
 * The HTTP server runs inside this process rather than as a child. A separate process
 * survives a force-quit of the window and leaves a port bound, which then makes the next
 * launch pick a different port or fail outright. Keeping it here means the operating
 * system reclaims everything together.
 *
 * Context isolation stays on and node integration stays off. This application renders
 * data pulled from public block explorers, which is untrusted input by definition, so the
 * page gets no direct access to the file system or to process spawning. What it needs
 * arrives through the explicit surface in preload.js.
 */

const { app, BrowserWindow, dialog, ipcMain, Menu, shell } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

const { createApp, listen, refresh, store } = require('../src/server/app');

const ROOT = path.resolve(__dirname, '..');
const IS_MAC = process.platform === 'darwin';

let win = null;
let server = null;
let port = null;

/* -------------------------------------------------------------------------- window --- */

function createWindow() {
  win = new BrowserWindow({
    width: 1560,
    height: 980,
    minWidth: 1100,
    minHeight: 700,
    backgroundColor: '#07090d',
    title: 'ChainTrace',
    show: false,
    autoHideMenuBar: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });

  win.loadURL(`http://127.0.0.1:${port}`);
  win.once('ready-to-show', () => win.show());
  win.on('closed', () => { win = null; });

  // An address or transaction id may link outward. None of it replaces this window.
  win.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith(`http://127.0.0.1:${port}`)) return { action: 'allow' };
    shell.openExternal(url);
    return { action: 'deny' };
  });
}

/* --------------------------------------------------------------------- background --- */

/**
 * Run one of the command-line stages and stream its output to the renderer.
 *
 * Ingestion and training both take minutes, so they cannot block the interface. The
 * window stays usable throughout and receives progress lines as they are produced.
 */
function runStage(name, command, args) {
  return new Promise((resolve) => {
    const send = (channel, payload) => {
      if (win && !win.isDestroyed()) win.webContents.send(channel, payload);
    };

    send('stage:start', { name });
    const proc = spawn(command, args, { cwd: ROOT, windowsHide: true });

    const relay = (buf) => {
      const text = buf.toString();
      for (const line of text.split(/\r?\n/)) {
        const trimmed = line.replace(/\r/g, '').trim();
        if (trimmed) send('stage:line', { name, line: trimmed });
      }
    };

    proc.stdout.on('data', relay);
    proc.stderr.on('data', relay);

    proc.on('error', (err) => {
      send('stage:done', { name, ok: false, error: err.message });
      resolve({ ok: false, error: err.message });
    });

    proc.on('exit', (code) => {
      // The server caches the window file by modification time, so a fresh ingest is
      // picked up on the next request without a restart.
      refresh();
      send('stage:done', { name, ok: code === 0, code });
      resolve({ ok: code === 0, code });
    });
  });
}

ipcMain.handle('stage:ingest', (event, opts = {}) => {
  const args = ['scripts/ingest.js'];
  if (opts.blocks) args.push('--blocks', String(opts.blocks));
  else args.push('--days', String(opts.days || 2));
  return runStage('ingest', process.execPath, args);
});

ipcMain.handle('stage:analyse', () =>
  runStage('analyse', opts_python(), ['python/pipeline.py']));

function opts_python() {
  return process.platform === 'win32' ? 'python' : 'python3';
}

ipcMain.handle('app:info', () => ({
  port,
  version: app.getVersion(),
  electron: process.versions.electron,
  node: process.versions.node,
  chrome: process.versions.chrome,
}));

ipcMain.handle('shell:openExternal', (event, url) => {
  if (typeof url === 'string' && /^https:\/\//.test(url)) shell.openExternal(url);
});

/* ---------------------------------------------------------------------------- menu --- */

function buildMenu() {
  const send = (channel, payload) => {
    if (win && !win.isDestroyed()) win.webContents.send(channel, payload);
  };

  Menu.setApplicationMenu(Menu.buildFromTemplate([
    ...(IS_MAC ? [{ role: 'appMenu' }] : []),
    {
      label: 'Data',
      submenu: [
        { label: 'Pull last 2 days', click: () => runStage('ingest', process.execPath, ['scripts/ingest.js', '--days', '2']) },
        { label: 'Pull last 12 hours', click: () => runStage('ingest', process.execPath, ['scripts/ingest.js', '--blocks', '72']) },
        { type: 'separator' },
        { label: 'Cluster and train', accelerator: 'CmdOrCtrl+T', click: () => runStage('analyse', opts_python(), ['python/pipeline.py']) },
        { type: 'separator' },
        { label: 'Reload view', accelerator: 'CmdOrCtrl+R', click: () => win && win.reload() },
        IS_MAC ? { role: 'close' } : { role: 'quit' },
      ],
    },
    {
      label: 'View',
      submenu: [
        { label: 'Triage', accelerator: 'CmdOrCtrl+1', click: () => send('nav', 'triage') },
        { label: 'Money flow', accelerator: 'CmdOrCtrl+2', click: () => send('nav', 'flow') },
        { label: 'Trace', accelerator: 'CmdOrCtrl+3', click: () => send('nav', 'trace') },
        { label: 'Cohorts', accelerator: 'CmdOrCtrl+4', click: () => send('nav', 'cohorts') },
        { label: 'Model', accelerator: 'CmdOrCtrl+5', click: () => send('nav', 'model') },
        { type: 'separator' },
        { role: 'toggleDevTools' },
        { role: 'resetZoom' }, { role: 'zoomIn' }, { role: 'zoomOut' },
        { type: 'separator' },
        { role: 'togglefullscreen' },
      ],
    },
  ]));
}

/* ----------------------------------------------------------------------- lifecycle --- */

// A second instance would bind the same port and fight over the same data files.
if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on('second-instance', () => {
    if (win) {
      if (win.isMinimized()) win.restore();
      win.focus();
    }
  });

  app.whenReady().then(async () => {
    buildMenu();
    try {
      const result = await listen(createApp(), 7400, '127.0.0.1', 20);
      server = result.server;
      port = result.port;
      refresh();
      createWindow();
    } catch (err) {
      dialog.showErrorBox(
        'ChainTrace could not start',
        `The local service failed to start:\n\n${err.message}\n\n` +
          'This usually means every port from 7400 to 7419 is already in use.'
      );
      app.quit();
    }

    app.on('activate', () => {
      if (BrowserWindow.getAllWindows().length === 0 && port) createWindow();
    });
  });

  app.on('window-all-closed', () => { if (!IS_MAC) app.quit(); });
  app.on('before-quit', () => { if (server) server.close(); });
}
