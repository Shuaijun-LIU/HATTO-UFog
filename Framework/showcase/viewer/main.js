/* Demo-style showcase viewer (Three.js r128, deterministic look).
 *
 * Inputs:
 * - world (Framework export): ../inputs/showcase_world.json
 * - trajectory:              ../inputs/showcase_trajectory.json
 *
 * Query params:
 * - camera=overview|chase|orbit|fpv|split
 * - external=overview|chase|orbit (only for camera=split)
 * - speed=<float> (playback multiplier)
 * - t0=<float> start time offset (seconds)
 * - duration=<float> mark done after t0+duration
 * - clarity=1 (disable fog/wind, reduce clutter for crisp capture)
 * - ui=0|1 (hide/show UI overlay)
 * - detail=0|1 (enable ground detail texture; off by default)
 * - roads=0|1 (enable road tinting on terrain; off by default)
 * - aa=0|1 (disable/enable WebGL antialias)
 * - fast=0|1 (use faster, less expensive shading)
 * - fixed_fps=<float> (optional fixed-step playback for capture)
 * - max_dt=<float> (clamp per-frame dt for smoother motion)
 * - fpv_roll=0|1 (roll FPV camera with UAV)
 * - fog=0|1 (fog is opt-in; default off)
 * - wind=0|1 (wind particles are opt-in; default off)
 * - shadows=0|1 (shadows are opt-in; default off)
 * - path=0|1 (trajectory path line is opt-in; default off)
 * - capture=0|1 (small presentation tweaks for automated capture)
 * - pdb=0|1 (preserveDrawingBuffer; off by default for performance)
 */

(() => {
  "use strict";

  const canvas = document.getElementById("c");
  const overlayEl = document.getElementById("overlay");
  const statusEl = document.getElementById("status");
  const cameraSelect = document.getElementById("cameraSelect");
  const btnPlay = document.getElementById("btnPlay");
  const btnReset = document.getElementById("btnReset");
  const speedRange = document.getElementById("speedRange");

  const urlParams = new URLSearchParams(window.location.search);
  const worldPath = urlParams.get("world") || "../inputs/showcase_world.json";
  const trajPath = urlParams.get("traj") || "../inputs/showcase_trajectory.json";
  const initialCamera = (urlParams.get("camera") || "overview").toLowerCase();
  const initialExternal = (urlParams.get("external") || "chase").toLowerCase();
  const t0 = Number.parseFloat(urlParams.get("t0") || "0");
  const durationLimit = urlParams.has("duration") ? Number.parseFloat(urlParams.get("duration")) : null;
  const clarityMode = urlParams.get("clarity") === "1";
  // Prefer "opt-in" visual clutter for presentation-grade defaults.
  const enableFog = !clarityMode && urlParams.get("fog") === "1";
  const enableWind = !clarityMode && urlParams.get("wind") === "1";
  const enableShadows = urlParams.get("shadows") === "1";
  const showPath = urlParams.get("path") === "1";
  const fpvRoll = urlParams.get("fpv_roll") === "1";
  const uiEnabled = urlParams.get("ui") !== "0";
  const enableDetail = !clarityMode && urlParams.get("detail") === "1";
  const enableRoads = urlParams.get("roads") === "1";
  const enableAA = urlParams.get("aa") !== "0";
  const fastMode = urlParams.get("fast") === "1";
  const fixedFpsRaw = Number.parseFloat(urlParams.get("fixed_fps") || "0");
  const maxDtRaw = urlParams.has("max_dt") ? Number.parseFloat(urlParams.get("max_dt") || "0") : Number.NaN;

  if (!uiEnabled && overlayEl) overlayEl.style.display = "none";

  function setStatus(text) {
    if (statusEl) statusEl.textContent = text;
  }

  function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
  }

  const fixedFps = Number.isFinite(fixedFpsRaw) && fixedFpsRaw > 0 ? clamp(fixedFpsRaw, 1, 240) : 0;
  const defaultMaxDt = clarityMode ? 1 / 25 : 0.05;
  const maxFrameDt = clamp(Number.isFinite(maxDtRaw) ? maxDtRaw : defaultMaxDt, 0.005, 0.2);

  const preferCaptureStyle = urlParams.get("capture") === "1";

  // Seeded RNG (Mulberry32) like the demo.
  let currentSeed = 1;
  function seedRandom(seed) {
    currentSeed = seed >>> 0;
  }
  function random() {
    let t = (currentSeed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  function parseFloatSafe(value, fallback) {
    const v = Number.parseFloat(value);
    return Number.isFinite(v) ? v : fallback;
  }

  function computeCumulativeDistances(points) {
    const s = new Float64Array(points.length);
    s[0] = 0;
    for (let i = 1; i < points.length; i += 1) {
      const dx = points[i].x - points[i - 1].x;
      const dy = points[i].y - points[i - 1].y;
      const dz = points[i].z - points[i - 1].z;
      s[i] = s[i - 1] + Math.sqrt(dx * dx + dy * dy + dz * dz);
    }
    return s;
  }

  function samplePath(points, cumDist, totalLen, d, outPos, outDir) {
    const pos = outPos || new THREE.Vector3();
    const dir = outDir || new THREE.Vector3(1, 0, 0);
    if (points.length < 2 || totalLen <= 1e-9) {
      if (points[0]) pos.copy(points[0]);
      else pos.set(0, 0, 0);
      dir.set(1, 0, 0);
      return;
    }
    const dist = ((d % totalLen) + totalLen) % totalLen;
    // Binary search
    let lo = 0;
    let hi = cumDist.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (cumDist[mid] < dist) lo = mid + 1;
      else hi = mid;
    }
    const i1 = clamp(lo, 1, cumDist.length - 1);
    const i0 = i1 - 1;
    const d0 = cumDist[i0];
    const d1 = cumDist[i1];
    const t = d1 <= d0 ? 0 : (dist - d0) / (d1 - d0);
    const p0 = points[i0];
    const p1 = points[i1];
    pos.lerpVectors(p0, p1, t);
    dir.subVectors(p1, p0);
    if (dir.lengthSq() > 1e-9) dir.normalize();
    else dir.set(1, 0, 0);
  }

  function makeDroneMesh(useLambert = false) {
    const group = new THREE.Group();

    const bodyMat = useLambert
      ? new THREE.MeshLambertMaterial({ color: 0x141a22 })
      : new THREE.MeshStandardMaterial({ color: 0x141a22, metalness: 0.4, roughness: 0.35 });
    const accentMat = useLambert
      ? new THREE.MeshLambertMaterial({ color: 0x38bdf8 })
      : new THREE.MeshStandardMaterial({ color: 0x38bdf8, metalness: 0.2, roughness: 0.25, emissive: 0x0b2a35 });
    const rotorMat = useLambert
      ? new THREE.MeshLambertMaterial({ color: 0x0b0f14 })
      : new THREE.MeshStandardMaterial({ color: 0x0b0f14, metalness: 0.05, roughness: 0.9 });

    const body = new THREE.Mesh(new THREE.BoxGeometry(2.2, 0.55, 1.0), bodyMat);
    body.castShadow = true;
    group.add(body);

    const nose = new THREE.Mesh(new THREE.BoxGeometry(0.45, 0.18, 0.35), accentMat);
    nose.position.set(1.2, 0.08, 0);
    nose.castShadow = true;
    group.add(nose);

    const armGeo = new THREE.BoxGeometry(3.6, 0.18, 0.18);
    const arm1 = new THREE.Mesh(armGeo, bodyMat);
    arm1.rotation.y = Math.PI / 4;
    arm1.castShadow = true;
    group.add(arm1);
    const arm2 = new THREE.Mesh(armGeo, bodyMat);
    arm2.rotation.y = -Math.PI / 4;
    arm2.castShadow = true;
    group.add(arm2);

    const rotorGeo = new THREE.CylinderGeometry(0.75, 0.75, 0.05, 14);
    const rotorOffsets = [
      [1.35, 0.28, 1.35],
      [1.35, 0.28, -1.35],
      [-1.35, 0.28, 1.35],
      [-1.35, 0.28, -1.35],
    ];
    for (const [x, y, z] of rotorOffsets) {
      const rotor = new THREE.Mesh(rotorGeo, rotorMat);
      rotor.position.set(x, y, z);
      rotor.castShadow = true;
      group.add(rotor);
    }

    group.userData.forwardAxis = new THREE.Vector3(1, 0, 0);
    return group;
  }

  function makeTextSprite(text, colorHex = "#ff4400") {
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    const fontSize = 42;
    ctx.font = `700 ${fontSize}px ui-monospace, Menlo, Consolas, monospace`;
    const padX = 14;
    const padY = 10;
    const metrics = ctx.measureText(text);
    const w = Math.ceil(metrics.width + padX * 2);
    const h = Math.ceil(fontSize + padY * 2);
    canvas.width = w;
    canvas.height = h;

    // Background
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "rgba(0,0,0,0.45)";
    ctx.strokeStyle = "rgba(255,255,255,0.22)";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.roundRect(2, 2, w - 4, h - 4, 10);
    ctx.fill();
    ctx.stroke();

    ctx.font = `700 ${fontSize}px ui-monospace, Menlo, Consolas, monospace`;
    ctx.fillStyle = colorHex;
    ctx.textBaseline = "middle";
    ctx.fillText(text, padX, h / 2);

    const tex = new THREE.CanvasTexture(canvas);
    tex.encoding = THREE.sRGBEncoding;
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false });
    const sprite = new THREE.Sprite(mat);
    sprite.scale.set(w / 45, h / 45, 1);
    sprite.renderOrder = 20;
    return sprite;
  }

  function heightSamplerFromWorld(world) {
    const hm = world.heightmap;
    if (!hm || !hm.heights || !hm.heights.length || !hm.heights[0].length) return null;
    const step = Number(hm.step_m || 1);
    const origin = hm.origin || [-(Number(hm.extent_m) || 1000) / 2, -(Number(hm.extent_m) || 1000) / 2];
    const heights = hm.heights;
    const rows = heights.length;
    const cols = heights[0].length;
    const ox = Number(origin[0] || 0);
    const oz = Number(origin[1] || 0);

    let minH = Infinity;
    let maxH = -Infinity;
    for (let r = 0; r < rows; r += 1) {
      const row = heights[r];
      for (let c = 0; c < cols; c += 1) {
        const v = Number(row[c] || 0);
        if (v < minH) minH = v;
        if (v > maxH) maxH = v;
      }
    }
    if (!Number.isFinite(minH)) minH = 0;
    if (!Number.isFinite(maxH)) maxH = 0;

    function sampleCore(x, z) {
      const fx = (x - ox) / step;
      const fz = (z - oz) / step;
      const ix = Math.floor(fx);
      const iz = Math.floor(fz);
      const tx = fx - ix;
      const tz = fz - iz;
      const x0 = clamp(ix, 0, cols - 1);
      const x1 = clamp(ix + 1, 0, cols - 1);
      const z0 = clamp(iz, 0, rows - 1);
      const z1 = clamp(iz + 1, 0, rows - 1);

      const h00 = Number(heights[z0][x0] || 0);
      const h10 = Number(heights[z0][x1] || 0);
      const h01 = Number(heights[z1][x0] || 0);
      const h11 = Number(heights[z1][x1] || 0);

      const hx0 = h00 * (1 - tx) + h10 * tx;
      const hx1 = h01 * (1 - tx) + h11 * tx;
      const h = hx0 * (1 - tz) + hx1 * tz;
      return { tx, tz, h, h00, h10, h01, h11, hx0, hx1 };
    }

    function sample(x, z) {
      return sampleCore(x, z).h;
    }

    function sampleWithGradient(x, z) {
      const s = sampleCore(x, z);
      const dhdx = ((s.h10 - s.h00) * (1 - s.tz) + (s.h11 - s.h01) * s.tz) / step;
      const dhdz = ((s.h01 - s.h00) * (1 - s.tx) + (s.h11 - s.h10) * s.tx) / step;
      return { h: s.h, gx: dhdx, gz: dhdz };
    }

    sample.minH = minH;
    sample.maxH = maxH;
    sample.step = step;
    sample.origin = { x: ox, z: oz };
    sample.sampleWithGradient = sampleWithGradient;

    return sample;
  }

  function buildTerrain(params) {
    const { mapSize, vertexCount, heightAt, lakes, simplex, isOnRoadFn, fastMode } = params;

    const geometry = new THREE.PlaneGeometry(mapSize, mapSize, vertexCount, vertexCount);
    geometry.rotateX(-Math.PI / 2);

    const positions = geometry.attributes.position;
    const colors = [];

    const cRoad = new THREE.Color(0x2a2f38);
    const cSand = new THREE.Color(0x9b8a6a);
    const cGrassLow = new THREE.Color(0x4f7f4f);
    const cGrassHigh = new THREE.Color(0x6b8f55);
    const cDirt = new THREE.Color(0x7b715d);
    const cRock = new THREE.Color(0x7c7c7c);
    const cSnow = new THREE.Color(0xe6eef7);

    const minH = Number.isFinite(heightAt.minH) ? Number(heightAt.minH) : -5;
    const maxH = Number.isFinite(heightAt.maxH) ? Number(heightAt.maxH) : 120;
    const denom = Math.max(1e-6, maxH - minH);

    function getLakeFactor(x, z, lake) {
      const dx = x - lake.x;
      const dz = z - lake.z;
      const cos = Math.cos(-lake.rot);
      const sin = Math.sin(-lake.rot);
      const nx = dx * cos - dz * sin;
      const nz = dx * sin + dz * cos;
      return (nx * nx) / (lake.rx * lake.rx) + (nz * nz) / (lake.rz * lake.rz);
    }

    function isNearLake(x, z) {
      for (const lake of lakes) {
        if (getLakeFactor(x, z, lake) < 1.15) return true;
      }
      return false;
    }

    for (let i = 0; i < positions.count; i += 1) {
      const x = positions.getX(i);
      const z = positions.getZ(i);
      const samp = heightAt.sampleWithGradient ? heightAt.sampleWithGradient(x, z) : { h: heightAt(x, z), gx: 0, gz: 0 };
      const h = samp.h;
      positions.setY(i, h);

      const isRoad = isOnRoadFn(x, z);
      const nearLake = isNearLake(x, z);
      const slope = Math.sqrt(samp.gx * samp.gx + samp.gz * samp.gz);
      const hNorm = clamp((h - minH) / denom, 0, 1);

      const micro = simplex.noise2D(x * 0.007, z * 0.007) * 0.5 + 0.5; // 0..1
      const dry = simplex.noise2D(x * 0.0015 + 100, z * 0.0015 + 100) * 0.5 + 0.5; // 0..1
      const shade = 0.92 + micro * 0.14;

      let r = 0;
      let g = 0;
      let b = 0;

      if (isRoad) {
        r = cRoad.r;
        g = cRoad.g;
        b = cRoad.b;
      } else if (nearLake && hNorm < 0.14) {
        r = cSand.r;
        g = cSand.g;
        b = cSand.b;
      } else if (hNorm > 0.86 || slope > 0.9) {
        r = cSnow.r;
        g = cSnow.g;
        b = cSnow.b;
      } else if (hNorm > 0.7 || slope > 0.55) {
        r = cRock.r;
        g = cRock.g;
        b = cRock.b;
      } else if (dry > 0.72 && hNorm < 0.55) {
        r = cDirt.r;
        g = cDirt.g;
        b = cDirt.b;
      } else {
        const t = clamp((hNorm - 0.12) / 0.55, 0, 1);
        r = cGrassLow.r * (1 - t) + cGrassHigh.r * t;
        g = cGrassLow.g * (1 - t) + cGrassHigh.g * t;
        b = cGrassLow.b * (1 - t) + cGrassHigh.b * t;
      }

      colors.push(r * shade, g * shade, b * shade);
    }

    geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
    geometry.computeVertexNormals();

    const mat = fastMode
      ? new THREE.MeshLambertMaterial({ vertexColors: true })
      : new THREE.MeshStandardMaterial({
          vertexColors: true,
          roughness: 0.9,
          metalness: 0.02,
        });
    const ground = new THREE.Mesh(geometry, mat);
    ground.receiveShadow = true;
    return ground;
  }

  function buildWater(lakes, heightAt, simplex, fastMode = false) {
    const group = new THREE.Group();
    const waterMat = fastMode
      ? new THREE.MeshLambertMaterial({ color: 0x2196f3, transparent: true, opacity: 0.82 })
      : new THREE.MeshStandardMaterial({ color: 0x2196f3, roughness: 0.12, transparent: true, opacity: 0.82 });

    function lakeWaterLevel(lake) {
      // Approximate a stable water level: min height on the ellipse boundary.
      let minH = Infinity;
      const samples = 24;
      for (let i = 0; i < samples; i += 1) {
        const a = (i / samples) * Math.PI * 2;
        const lx = lake.x + Math.cos(a) * lake.rx * 0.92;
        const lz = lake.z + Math.sin(a) * lake.rz * 0.92;
        const h = heightAt(lx, lz);
        if (h < minH) minH = h;
      }
      if (!Number.isFinite(minH)) minH = 0;
      return minH + 0.18;
    }

    for (const lake of lakes) {
      const waterGeo = new THREE.CircleGeometry(1, 64);
      const wPos = waterGeo.attributes.position;
      for (let i = 1; i < wPos.count; i += 1) {
        const x = wPos.getX(i);
        const y = wPos.getY(i);
        const angle = Math.atan2(y, x);
        const rNoise = simplex.noise2D(Math.cos(angle) * 2, Math.sin(angle) * 2);
        const scale = 1.0 + rNoise * 0.1;
        wPos.setX(i, x * scale);
        wPos.setY(i, y * scale);
      }
      waterGeo.rotateX(-Math.PI / 2);
      waterGeo.scale(lake.rx, 1, lake.rz);
      waterGeo.rotateY(-lake.rot);
      const water = new THREE.Mesh(waterGeo, waterMat);
      water.position.set(lake.x, lakeWaterLevel(lake), lake.z);
      water.receiveShadow = false;
      water.renderOrder = 2;
      group.add(water);
    }
    return group;
  }

  function buildBuildings(buildings, heightAt, windowTex = null, fastMode = false) {
    if (!buildings || !buildings.length) return null;
    const geom = new THREE.BoxGeometry(1, 1, 1);
    geom.translate(0, 0.5, 0);
    const mat = fastMode
      ? new THREE.MeshLambertMaterial({
          color: 0xffffff,
          map: windowTex || undefined,
        })
      : new THREE.MeshStandardMaterial({
          color: 0xffffff,
          roughness: 0.84,
          metalness: 0.04,
          map: windowTex || undefined,
          emissive: windowTex ? new THREE.Color(0xffffff) : new THREE.Color(0x000000),
          emissiveMap: windowTex || undefined,
          emissiveIntensity: windowTex ? 0.12 : 0.0,
        });
    const mesh = new THREE.InstancedMesh(geom, mat, buildings.length);
    mesh.castShadow = true;
    mesh.receiveShadow = true;

    const dummy = new THREE.Object3D();
    for (let i = 0; i < buildings.length; i += 1) {
      const b = buildings[i];
      const x = Number(b.x);
      const z = Number(b.y);
      const width = Number(b.width);
      const depth = Number(b.depth);
      const baseZ = Number(b.base_z);
      const height = Number(b.height);

      // Use base_z but extend downward to avoid any floating artifacts.
      const visualBase = baseZ - 80;
      const top = baseZ + height;
      const h = Math.max(1.0, top - visualBase);

      dummy.position.set(x, visualBase, z);
      dummy.scale.set(width, h, depth);
      dummy.rotation.y = (i * 0.17) % (Math.PI * 0.12);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);

      // Slight per-building color variation (subtle, more realistic).
      if (mesh.setColorAt) {
        const jitter = 0.06 * ((i * 97) % 11) / 10 - 0.03;
        const c = new THREE.Color(0xd6dbe3);
        c.offsetHSL(0, 0, jitter);
        mesh.setColorAt(i, c);
      }
    }
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.instanceMatrix.needsUpdate = true;
    return mesh;
  }

  function buildTrees(params) {
    const { mapSize, heightAt, lakes, isOnRoadFn, count, seed, fastMode } = params;
    seedRandom(seed ^ 0xa5a5a5a5);

    function getLakeFactor(x, z, lake) {
      const dx = x - lake.x;
      const dz = z - lake.z;
      const cos = Math.cos(-lake.rot);
      const sin = Math.sin(-lake.rot);
      const nx = dx * cos - dz * sin;
      const nz = dx * sin + dz * cos;
      return (nx * nx) / (lake.rx * lake.rx) + (nz * nz) / (lake.rz * lake.rz);
    }
    function isInLake(x, z, buffer = 0) {
      for (const lake of lakes) {
        const f = getLakeFactor(x, z, lake);
        if (f < 1.0 + buffer / lake.rx) return true;
      }
      return false;
    }

    const treeGeo = new THREE.ConeGeometry(2, 6, 6);
    treeGeo.translate(0, 3, 0);
    const treeMat = fastMode
      ? new THREE.MeshLambertMaterial({ color: 0x1b5e20, flatShading: true })
      : new THREE.MeshStandardMaterial({ color: 0x1b5e20, roughness: 0.92, flatShading: true });
    const trees = new THREE.InstancedMesh(treeGeo, treeMat, count);
    trees.castShadow = true;
    trees.receiveShadow = false;

    const dummy = new THREE.Object3D();
    let tCount = 0;
    let attempts = 0;
    while (tCount < count && attempts < count * 12) {
      attempts += 1;
      const x = (random() - 0.5) * mapSize;
      const z = (random() - 0.5) * mapSize;
      if (isInLake(x, z, 6)) continue;
      if (isOnRoadFn(x, z)) continue;
      const y = heightAt(x, z);
      if (y < -5) continue;
      dummy.position.set(x, y, z);
      const s = 0.65 + random() * 0.6;
      dummy.scale.set(s, s, s);
      dummy.rotation.y = random() * Math.PI;
      dummy.updateMatrix();
      trees.setMatrixAt(tCount, dummy.matrix);
      tCount += 1;
    }
    trees.count = tCount;
    trees.instanceMatrix.needsUpdate = true;
    return trees;
  }

  function makeGroundDetailTexture(seed) {
    const savedSeed = currentSeed;
    seedRandom(seed ^ 0x9e3779b9);
    const size = 256;
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(size, size);
    for (let i = 0; i < img.data.length; i += 4) {
      const n = 0.78 + random() * 0.22;
      img.data[i] = Math.round(255 * n);
      img.data[i + 1] = Math.round(255 * (n * 1.0));
      img.data[i + 2] = Math.round(255 * (n * 0.98));
      img.data[i + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    const tex = new THREE.CanvasTexture(canvas);
    tex.encoding = THREE.sRGBEncoding;
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    tex.minFilter = THREE.LinearMipMapLinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.anisotropy = 8;
    seedRandom(savedSeed);
    return tex;
  }

  function makeBuildingWindowTexture(seed) {
    const savedSeed = currentSeed;
    seedRandom(seed ^ 0x243f6a88);
    // Higher-resolution window texture so 4K captures don't look blocky.
    const w = 1024;
    const h = 2048;
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");

    // Base facade gradient (subtle variation to avoid a flat "grid" look).
    const bg = ctx.createLinearGradient(0, 0, 0, h);
    bg.addColorStop(0.0, "#e3e7ef");
    bg.addColorStop(0.55, "#d5dbe5");
    bg.addColorStop(1.0, "#cfd6e0");
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, w, h);

    // Lower-frequency window pattern (reduces moiré at distance).
    const cols = 18;
    const rows = 48;
    const padX = 20;
    const padY = 32;
    const cellW = (w - padX * 2) / cols;
    const cellH = (h - padY * 2) / rows;
    const colJitter = new Array(cols).fill(0).map(() => (random() - 0.5) * 0.08);
    for (let r = 0; r < rows; r += 1) {
      for (let c = 0; c < cols; c += 1) {
        const lit = random() < 0.12;
        const x = Math.round(padX + c * cellW + cellW * (0.22 + colJitter[c]));
        const y = Math.round(padY + r * cellH + cellH * 0.18);
        const ww = Math.max(2, Math.round(cellW * (0.56 + (random() - 0.5) * 0.06)));
        const hh = Math.max(2, Math.round(cellH * (0.58 + (random() - 0.5) * 0.06)));
        if (lit) ctx.fillStyle = "rgba(255, 225, 160, 0.78)";
        else ctx.fillStyle = "rgba(18, 24, 34, 0.22)";
        ctx.fillRect(x, y, ww, hh);
      }
    }

    // Subtle facade noise to break up large flat regions.
    ctx.fillStyle = "rgba(0,0,0,0.035)";
    for (let i = 0; i < 14000; i += 1) {
      const x = Math.floor(random() * w);
      const y = Math.floor(random() * h);
      const a = 0.02 + random() * 0.04;
      ctx.fillStyle = `rgba(0,0,0,${a.toFixed(3)})`;
      ctx.fillRect(x, y, 1, 1);
    }

    const grad = ctx.createLinearGradient(0, 0, w, 0);
    grad.addColorStop(0, "rgba(0,0,0,0.06)");
    grad.addColorStop(0.5, "rgba(0,0,0,0.00)");
    grad.addColorStop(1, "rgba(0,0,0,0.07)");
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, w, h);

    const tex = new THREE.CanvasTexture(canvas);
    tex.encoding = THREE.sRGBEncoding;
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(1, 1);
    tex.minFilter = THREE.LinearMipMapLinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.anisotropy = 16;
    seedRandom(savedSeed);
    return tex;
  }

  function makeGlowSprite(colorHex = "#38bdf8") {
    const size = 192;
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext("2d");
    const cx = size / 2;
    const cy = size / 2;
    const r = size / 2;
    const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
    grad.addColorStop(0.0, `${colorHex}cc`);
    grad.addColorStop(0.35, `${colorHex}66`);
    grad.addColorStop(1.0, `${colorHex}00`);
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, size, size);

    const tex = new THREE.CanvasTexture(canvas);
    tex.encoding = THREE.sRGBEncoding;
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    const mat = new THREE.SpriteMaterial({
      map: tex,
      transparent: true,
      depthTest: false,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    const sprite = new THREE.Sprite(mat);
    sprite.renderOrder = 30;
    return sprite;
  }

  // (Optional) FPV overlay can be added later; keep the viewer minimal for clean capture.

  function makeSkyDome() {
    const canvas = document.createElement("canvas");
    canvas.width = 512;
    canvas.height = 512;
    const ctx = canvas.getContext("2d");

    const grad = ctx.createLinearGradient(0, 0, 0, canvas.height);
    grad.addColorStop(0.0, "#5fb4ff");
    grad.addColorStop(0.5, "#bfe4ff");
    grad.addColorStop(1.0, "#edf6ff");
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Subtle sun glow.
    const sx = canvas.width * 0.78;
    const sy = canvas.height * 0.23;
    const sun = ctx.createRadialGradient(sx, sy, 0, sx, sy, canvas.width * 0.22);
    sun.addColorStop(0.0, "rgba(255,255,255,0.65)");
    sun.addColorStop(0.35, "rgba(255,255,255,0.25)");
    sun.addColorStop(1.0, "rgba(255,255,255,0.0)");
    ctx.fillStyle = sun;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const tex = new THREE.CanvasTexture(canvas);
    tex.encoding = THREE.sRGBEncoding;
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;

    const geo = new THREE.SphereGeometry(5200, 32, 18);
    const mat = new THREE.MeshBasicMaterial({
      map: tex,
      side: THREE.BackSide,
      depthWrite: false,
      fog: false,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.renderOrder = -10;
    return mesh;
  }

  function buildWindParticles(params) {
    const { mapSize } = params;
    const particleCount = 1200;
    const particleGeo = new THREE.BufferGeometry();
    const pPositions = new Float32Array(particleCount * 3);
    const pVel = new Float32Array(particleCount);
    for (let i = 0; i < particleCount; i += 1) {
      pPositions[i * 3] = (Math.random() - 0.5) * mapSize * 0.55;
      pPositions[i * 3 + 1] = (Math.random() - 0.5) * 220 + 110;
      pPositions[i * 3 + 2] = (Math.random() - 0.5) * mapSize * 0.55;
      pVel[i] = 0.55 + Math.random();
    }
    particleGeo.setAttribute("position", new THREE.BufferAttribute(pPositions, 3));
    const mat = new THREE.PointsMaterial({
      color: 0xcff7ff,
      size: 0.6,
      transparent: true,
      opacity: 0.14,
      sizeAttenuation: true,
      depthWrite: false,
    });
    const pts = new THREE.Points(particleGeo, mat);
    pts.userData.vel = pVel;
    pts.renderOrder = 1;
    return pts;
  }

  function addNfzMarkers(scene, obstacles, heightAt) {
    if (!obstacles || obstacles.length === 0) return;
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0xff4400,
      transparent: true,
      opacity: 0.14,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    const lineMat = new THREE.LineDashedMaterial({
      color: 0xffaa00,
      transparent: true,
      opacity: 0.55,
      dashSize: 4,
      gapSize: 3,
      depthWrite: false,
    });

    for (let i = 0; i < obstacles.length; i += 1) {
      const o = obstacles[i];
      const x = Number(o.x);
      const z = Number(o.y);
      const r = Math.max(2.0, Number(o.radius || 10));
      const y0 = heightAt(x, z) + 0.25;

      const ringGeo = new THREE.RingGeometry(r * 0.92, r, 72);
      ringGeo.rotateX(-Math.PI / 2);
      const ring = new THREE.Mesh(ringGeo, ringMat);
      ring.position.set(x, y0, z);
      ring.renderOrder = 5;
      scene.add(ring);

      const y1 = y0 + 90;
      const lineGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(x, y0, z), new THREE.Vector3(x, y1, z)]);
      const line = new THREE.Line(lineGeo, lineMat);
      line.computeLineDistances();
      line.renderOrder = 6;
      scene.add(line);

      const label = makeTextSprite(`NFZ-${i + 1}`);
      label.position.set(x, y1 + 6, z);
      scene.add(label);
    }
  }

  async function loadJson(path) {
    const res = await fetch(path, { cache: "no-store" });
    if (!res.ok) throw new Error(`Failed to load ${path}: ${res.status}`);
    return await res.json();
  }

  async function main() {
    if (!window.THREE || !window.SimplexNoise) {
      setStatus("Missing THREE / SimplexNoise (CDN load failed).");
      return;
    }

    setStatus("Loading inputs…");
    const [world, traj] = await Promise.all([loadJson(worldPath), loadJson(trajPath)]);

    const worldCfg = world.config || {};
    const mapSize = Number(worldCfg.size_m || 1000);
    const mapHeight = Number(worldCfg.height_m || 300);
    const seed = Number(worldCfg.seed || 1234);

    seedRandom(seed);
    const simplex = new SimplexNoise(random);

    const heightAt = heightSamplerFromWorld(world);
    if (!heightAt) {
      setStatus("Missing heightmap in world.json (enable world.heightmap_step_m).");
      return;
    }

    // Camera / scene / renderer (demo-like baseline + small upgrades)
    const scene = new THREE.Scene();
    const fogColor = 0x9fc7ea;
    scene.background = new THREE.Color(fogColor);
    scene.fog = enableFog ? new THREE.FogExp2(fogColor, 0.0011) : null;
    scene.add(makeSkyDome());

    const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 6000);
    camera.position.set(0, 220, 360);
    const fpvCamera = new THREE.PerspectiveCamera(72, 1, 0.05, 6000);
    fpvCamera.up.set(0, 1, 0);

    const preserveBuffer = urlParams.get("pdb") === "1";
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: enableAA, preserveDrawingBuffer: preserveBuffer });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
    renderer.shadowMap.enabled = enableShadows;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.outputEncoding = THREE.sRGBEncoding;
    renderer.physicallyCorrectLights = !fastMode;
    // Keep filmic tone mapping even in fast mode to avoid washed-out output.
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = fastMode ? 0.82 : 0.95;
    renderer.autoClear = false;

    const hemiLight = new THREE.HemisphereLight(0xffffff, 0x3a4a3a, fastMode ? 0.6 : 0.72);
    scene.add(hemiLight);
    const dirLight = new THREE.DirectionalLight(0xffffff, fastMode ? 0.7 : 0.85);
    dirLight.position.set(160, 330, 120);
    dirLight.castShadow = enableShadows;
    dirLight.shadow.mapSize.width = 4096;
    dirLight.shadow.mapSize.height = 4096;
    dirLight.shadow.camera.left = -650;
    dirLight.shadow.camera.right = 650;
    dirLight.shadow.camera.top = 650;
    dirLight.shadow.camera.bottom = -650;
    scene.add(dirLight);

    // Camera controls (interactive convenience only)
    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    // Disable by default to avoid interfering with the scripted camera modes.
    controls.enabled = false;

    // Demo-like city boundary and roads (used only for vertex colors and tree placement)
    function getCityLimit(angle) {
      const noise = simplex.noise2D(Math.cos(angle), Math.sin(angle));
      return 160 + noise * 70;
    }
    function isOnRoad(x, z) {
      const dist = Math.sqrt(x * x + z * z);
      const angle = Math.atan2(z, x);
      const limit = getCityLimit(angle);
      if (dist > limit - 10) return false;
      if (Math.abs(z - x) < 7 && dist > 60) return true;
      if (Math.abs(x - Math.sin(z * 0.05) * 22) < 7 && z < -60) return true;
      if (Math.abs(z - Math.sin(x * 0.03) * 32 - 20) < 7 && x > 60) return true;
      const ringNoise = simplex.noise2D(x * 0.01, z * 0.01) * 22;
      if (Math.abs(dist - (125 + ringNoise)) < 6) return true;
      return false;
    }
    const isOnRoadFn = enableRoads ? isOnRoad : () => false;

    // Lakes (map Framework coords x/y -> demo coords x/z)
    const lakes = (world.lakes || []).map((l) => ({
      x: Number(l.cx),
      z: Number(l.cy),
      rx: Math.max(10, Number(l.rx)),
      rz: Math.max(10, Number(l.ry)),
      rot: Number(l.rot || 0),
    }));

    // Terrain mesh resolution is a major performance lever.
    // - Default: close to the heightmap sampling resolution.
    // - fast=1: downsampled mesh (still visually plausible) for smoother capture in CPU-only renderers.
    const heightStep = Number.isFinite(heightAt.step) ? Number(heightAt.step) : 2.0;
    const baseVertexCount = Math.round(mapSize / Math.max(1e-6, heightStep));
    const vMin = fastMode ? 180 : 420;
    const vMax = fastMode ? 320 : 700;
    const vertexCount = clamp(Math.round(baseVertexCount * (fastMode ? 0.35 : 1.0)), vMin, vMax);
	    const ground = buildTerrain({
	      mapSize,
	      vertexCount,
	      heightAt,
	      lakes,
	      simplex,
	      isOnRoadFn,
	      fastMode,
	    });
    if (enableDetail) {
      const groundDetail = makeGroundDetailTexture(seed);
      groundDetail.repeat.set(mapSize / 160, mapSize / 160);
      ground.material.map = groundDetail;
      ground.material.needsUpdate = true;
    }
    scene.add(ground);

    const water = buildWater(lakes, heightAt, simplex, fastMode);
    scene.add(water);

    const buildingTex = makeBuildingWindowTexture(seed);
    const buildings = buildBuildings(world.buildings || [], heightAt, buildingTex, fastMode);
    if (buildings) scene.add(buildings);

    const treeCount = clarityMode || fastMode ? 1400 : 3000;
	    const trees = buildTrees({ mapSize, heightAt, lakes, isOnRoadFn, count: treeCount, seed, fastMode });
	    scene.add(trees);

    const wind = enableWind ? buildWindParticles({ mapSize }) : null;
    if (wind) scene.add(wind);

	    addNfzMarkers(scene, world.obstacles || [], heightAt);

	    // Drone + trajectory
	    const drone = makeDroneMesh(fastMode);
    drone.scale.setScalar(1.9);
    const droneGlow = makeGlowSprite("#38bdf8");
    droneGlow.position.set(0, 3.0, 0);
    droneGlow.scale.set(preferCaptureStyle ? 12 : 16, preferCaptureStyle ? 12 : 16, 1);
    drone.add(droneGlow);
    scene.add(drone);

    const rawPath = (traj.path || []).map((p) => {
      // sim(x,y,z) -> demo (x,y,z) where y is up and z is horizontal
      return new THREE.Vector3(Number(p[0]), Number(p[2]), Number(p[1]));
    });
    const speedMps = parseFloatSafe(traj.meta?.speed_m_s, 14.0);
    const path = rawPath.filter((v) => Number.isFinite(v.x) && Number.isFinite(v.y) && Number.isFinite(v.z));
    const cumDist = computeCumulativeDistances(path);
    const totalLen = cumDist.length ? cumDist[cumDist.length - 1] : 0;

    const pathLine = showPath
      ? new THREE.Line(
          new THREE.BufferGeometry().setFromPoints(path.slice(0, Math.min(5000, path.length))),
          new THREE.LineBasicMaterial({ color: 0x0ea5e9, transparent: true, opacity: 0.55 }),
        )
      : null;
    if (pathLine) {
      pathLine.renderOrder = 4;
      scene.add(pathLine);
    }

    // Playback state
    let playing = urlParams.get("autoplay") !== "0";
    let speedScale = clamp(parseFloatSafe(urlParams.get("speed"), 1.0), 0.05, 5.0);
    let simTime = t0;
    let lastFrameMs = performance.now();
    const allowedCameraModes = ["overview", "chase", "orbit", "fpv", "split"];
    let cameraMode = allowedCameraModes.includes(initialCamera) ? initialCamera : "overview";
    let lastSingleMode = ["overview", "chase", "orbit", "fpv"].includes(cameraMode)
      ? cameraMode
      : ["overview", "chase", "orbit"].includes(initialExternal)
        ? initialExternal
        : "chase";
    let splitExternalMode = ["overview", "chase", "orbit"].includes(initialExternal) ? initialExternal : lastSingleMode;

    if (speedRange) speedRange.value = String(speedScale);
    if (cameraSelect) cameraSelect.value = cameraMode;
    if (btnPlay) btnPlay.textContent = playing ? "Pause" : "Play";

    function getTimeSec() {
      return simTime;
    }

    function setCameraMode(mode) {
      const m = allowedCameraModes.includes(mode) ? mode : "overview";
      if (m !== "split" && ["overview", "chase", "orbit"].includes(m)) lastSingleMode = m;
      if (m === "split") splitExternalMode = lastSingleMode;
      cameraMode = m;
      if (cameraSelect) cameraSelect.value = m;
      resize();
    }
    function resetTime() {
      simTime = t0;
      lastFrameMs = performance.now();
    }
    function seekTime(t) {
      simTime = Number.isFinite(t) ? t : 0;
      lastFrameMs = performance.now();
    }

    // Capture hooks for headless tooling
    window.__SHOWCASE_READY = true;
    window.__SHOWCASE_DONE = false;
    window.__SHOWCASE_SET_CAMERA = setCameraMode;
    window.__SHOWCASE_SEEK = seekTime;
    window.__SHOWCASE_RESET = resetTime;
    window.__SHOWCASE_RECORD = async (opts) => {
      const o = opts && typeof opts === "object" ? opts : {};
      const filename = typeof o.filename === "string" && o.filename ? o.filename : "showcase_recording.webm";
      const durationS = clamp(parseFloatSafe(o.durationS, 8.0), 0.2, 1200);
      const fps = clamp(parseFloatSafe(o.fps, 25.0), 1, 120);
      const bitrate = clamp(parseFloatSafe(o.bitrate, 24_000_000), 200_000, 200_000_000);

      if (!canvas || typeof canvas.captureStream !== "function" || typeof MediaRecorder === "undefined") {
        throw new Error("MediaRecorder/captureStream not supported in this browser.");
      }

      // Ensure the renderer has presented a stable frame before recording.
      await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));

      const stream = canvas.captureStream(fps);
      const mimeCandidates = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"];
      let mimeType = "";
      for (const m of mimeCandidates) {
        if (MediaRecorder.isTypeSupported(m)) {
          mimeType = m;
          break;
        }
      }

      const chunks = [];
      const recorder = new MediaRecorder(stream, { mimeType, videoBitsPerSecond: bitrate });
      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size) chunks.push(e.data);
      };

      const done = new Promise((resolve, reject) => {
        recorder.onerror = (evt) => reject(evt?.error || new Error("MediaRecorder error"));
        recorder.onstop = () => resolve();
      });

      recorder.start(250);
      if (!playing && btnPlay) btnPlay.click();
      await new Promise((r) => setTimeout(r, durationS * 1000));
      recorder.stop();
      await done;

      const blob = new Blob(chunks, { type: recorder.mimeType || mimeType || "video/webm" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      a.style.display = "none";
      document.body.appendChild(a);
      a.click();
      setTimeout(() => {
        URL.revokeObjectURL(url);
        a.remove();
      }, 1200);
      return { filename, bytes: blob.size, mime: blob.type, fps, bitrate };
    };

    if (btnPlay) {
      btnPlay.addEventListener("click", () => {
        playing = !playing;
        lastFrameMs = performance.now();
        btnPlay.textContent = playing ? "Pause" : "Play";
      });
    }
    if (btnReset) btnReset.addEventListener("click", () => resetTime());
    if (cameraSelect) cameraSelect.addEventListener("change", () => setCameraMode(cameraSelect.value));
    if (speedRange) {
      speedRange.addEventListener("input", () => {
        speedScale = clamp(parseFloatSafe(speedRange.value, 1.0), 0.05, 5.0);
        lastFrameMs = performance.now();
      });
    }

    function resize() {
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      renderer.setSize(w, h, false);
      const denom = Math.max(1, h);
      const aspectFull = w / denom;
      const aspectHalf = Math.max(1, w / 2) / denom;
      camera.aspect = cameraMode === "split" ? aspectHalf : aspectFull;
      camera.updateProjectionMatrix();
      fpvCamera.aspect = cameraMode === "split" ? aspectHalf : aspectFull;
      fpvCamera.updateProjectionMatrix();
    }
    window.addEventListener("resize", resize);
    resize();

    const forwardAxis = drone.userData.forwardAxis || new THREE.Vector3(1, 0, 0);
    let lastPos = path.length ? path[0].clone() : new THREE.Vector3();
    const upVec = new THREE.Vector3(0, 1, 0);
    const tmpVecA = new THREE.Vector3();
    const tmpVecB = new THREE.Vector3();
    const tmpVecC = new THREE.Vector3();
    const pathPos = new THREE.Vector3();
    const pathDir = new THREE.Vector3(1, 0, 0);
    const tmpQuatA = new THREE.Quaternion();
    const renderSize = new THREE.Vector2();
    const centerTarget = new THREE.Vector3(0, 40, 0);
    const overviewPos = new THREE.Vector3(0.35 * mapSize, 0.8 * mapHeight, 0.55 * mapSize);
    const fpvAdjust = new THREE.Quaternion().setFromAxisAngle(upVec, -Math.PI / 2);

    function updateCamera(uavPos, dir, mode) {
      if (mode === "overview") {
        camera.position.lerp(overviewPos, 0.06);
        camera.lookAt(centerTarget);
        return;
      }
      if (mode === "orbit") {
        const t = getTimeSec();
        const r = 0.65 * mapSize;
        const ang = t * 0.09;
        tmpVecA.set(Math.cos(ang) * r, 0.72 * mapHeight, Math.sin(ang) * r);
        camera.position.lerp(tmpVecA, 0.05);
        camera.lookAt(centerTarget);
        return;
      }
      // chase
      tmpVecA.copy(dir).multiplyScalar(-62).addScaledVector(upVec, 26).add(uavPos);
      camera.position.lerp(tmpVecA, 0.16);
      camera.lookAt(uavPos);
    }

    function animate() {
      const now = performance.now();
      let dt = (now - lastFrameMs) / 1000;
      lastFrameMs = now;
      if (playing) {
        if (fixedFps > 0) dt = 1 / fixedFps;
        dt = clamp(dt, 0, maxFrameDt);
        simTime += dt * speedScale;
      }
      const t = getTimeSec();
      const dist = t * speedMps;
      samplePath(path, cumDist, totalLen, dist, pathPos, pathDir);

      drone.position.copy(pathPos);
      tmpQuatA.setFromUnitVectors(forwardAxis, pathDir.lengthSq() > 1e-9 ? pathDir : forwardAxis);
      drone.quaternion.slerp(tmpQuatA, 0.18);

      // Subtle bank for motion clarity.
      tmpVecA.subVectors(pathPos, lastPos);
      const bank = clamp(tmpVecA.length() * 0.03, 0, 0.22);
      drone.rotation.z = -bank;
      lastPos.copy(pathPos);

      // FPV camera update (stable horizon by default; optional roll).
      fpvCamera.position.copy(pathPos).addScaledVector(upVec, 1.4).addScaledVector(pathDir, 0.8);
      if (fpvRoll) {
        fpvCamera.quaternion.copy(drone.quaternion).multiply(fpvAdjust);
      } else {
        // Keep FPV view readable: lock to a forward-horizontal direction with a slight downward pitch.
        tmpVecC.copy(pathDir);
        tmpVecC.y = 0;
        if (tmpVecC.lengthSq() < 1e-9) tmpVecC.set(1, 0, 0);
        tmpVecC.normalize();
        // More downward pitch so the FPV panel includes terrain/buildings (less empty sky).
        tmpVecC.y = -0.18;
        tmpVecC.normalize();
        tmpVecB.copy(pathPos).addScaledVector(tmpVecC, 260);
        fpvCamera.lookAt(tmpVecB);
      }

      // Wind particles drift
      if (wind && wind.geometry && wind.geometry.attributes && wind.geometry.attributes.position) {
        const p = wind.geometry.attributes.position;
        const v = wind.userData.vel;
        for (let i = 0; i < p.count; i += 1) {
          p.setX(i, p.getX(i) + v[i] * 0.45);
          if (p.getX(i) > 0.55 * mapSize) p.setX(i, -0.55 * mapSize);
        }
        p.needsUpdate = true;
      }

      const externalMode = cameraMode === "split" ? splitExternalMode : cameraMode;
      if (externalMode !== "fpv") updateCamera(pathPos, pathDir, externalMode);
      if (controls.enabled) controls.update();

      renderer.getSize(renderSize);
      const w = renderSize.x;
      const h = renderSize.y;
      renderer.setViewport(0, 0, w, h);
      renderer.setScissorTest(false);
      renderer.clear(true, true, true);

      if (cameraMode === "split") {
        const half = Math.floor(w / 2);
        renderer.setScissorTest(true);
        renderer.setViewport(0, 0, half, h);
        renderer.setScissor(0, 0, half, h);
        drone.visible = true;
        renderer.render(scene, camera);
        renderer.setViewport(half, 0, w - half, h);
        renderer.setScissor(half, 0, w - half, h);
        // Hide the UAV model in FPV for a cleaner "camera feed" look.
        drone.visible = false;
        renderer.render(scene, fpvCamera);
        drone.visible = true;
        renderer.setScissorTest(false);
      } else if (cameraMode === "fpv") {
        drone.visible = false;
        renderer.render(scene, fpvCamera);
        drone.visible = true;
      } else {
        renderer.render(scene, camera);
      }

      if (durationLimit !== null && t >= t0 + durationLimit) {
        window.__SHOWCASE_DONE = true;
      }
    }

    setStatus(`Loaded. size=${mapSize.toFixed(0)}m, path=${path.length} pts`);
    renderer.setAnimationLoop(animate);
  }

  main().catch((err) => {
    console.error(err);
    setStatus(String(err));
  });
})();
