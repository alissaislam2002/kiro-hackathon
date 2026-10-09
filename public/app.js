import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { FBXLoader } from "three/addons/loaders/FBXLoader.js";
import { VRMLoaderPlugin, VRMUtils } from "@pixiv/three-vrm";

const $ = (selector) => document.querySelector(selector);
const canvas = $("#avatar-canvas");
const startScreen = $("#start-screen");
const startButton = $("#start-button");
const loadState = $("#load-state");
const bubble = $("#speech-bubble");
const transcript = $("#transcript");
const composer = $("#composer");
const messageInput = $("#message-input");
const sendButton = $("#send-button");
const count = $("#character-count");
const taskForm = $("#task-form");
const taskTitle = $("#task-title");
const taskDue = $("#task-due");
const taskList = $("#task-list");
const taskTemplate = $("#task-template");
const summaryButton = $("#summary-button");
const danceButton = $("#dance-button");
const micButton = $("#mic-button");
const voiceToggle = $("#voice-toggle");
const serviceState = $("#service-state");
const conversationLog = $("#conversation-log");

let socket;
let connected = false;
let started = false;
let busy = false;
let voiceEnabled = true;
let tasks = [];
let serviceStatus = {};
let reconnectTimer;
const clientId = localStorage.getItem("nour-client-id") || crypto.randomUUID();
localStorage.setItem("nour-client-id", clientId);

const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.shadowMap.enabled = true;
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 20);
camera.position.set(0, 1.0, 4.0);
camera.lookAt(0, 0.75, 0);
scene.add(new THREE.HemisphereLight(0xd9fff4, 0x244254, 2.1));
const keyLight = new THREE.DirectionalLight(0xffe6bd, 2.7);
keyLight.position.set(-2, 4, 3);
scene.add(keyLight);
const rimLight = new THREE.DirectionalLight(0x61d9d4, 2.2);
rimLight.position.set(3, 2, -2);
scene.add(rimLight);

let vrm;
let mixer;
let currentAction;
let currentAnimation = "idle";
let modelReadyResolve;
const modelReady = new Promise((resolve) => { modelReadyResolve = resolve; });
const clips = new Map();
const clock = new THREE.Clock();
let blinkAt = performance.now() + 1800;
let blinkStarted = 0;
let lipSync = null;
let fallbackMouth = false;
let emotion = "neutral";
let danceAudio;

const boneMap = {
  Hips: "hips", Spine: "spine", Spine1: "chest", Spine2: "upperChest", Neck: "neck", Head: "head",
  LeftShoulder: "leftShoulder", LeftArm: "leftUpperArm", LeftForeArm: "leftLowerArm", LeftHand: "leftHand",
  LeftUpLeg: "leftUpperLeg", LeftLeg: "leftLowerLeg", LeftFoot: "leftFoot", LeftToeBase: "leftToes",
  RightShoulder: "rightShoulder", RightArm: "rightUpperArm", RightForeArm: "rightLowerArm", RightHand: "rightHand",
  RightUpLeg: "rightUpperLeg", RightLeg: "rightLowerLeg", RightFoot: "rightFoot", RightToeBase: "rightToes"
};
for (const side of ["Left", "Right"]) {
  const lower = side.toLowerCase();
  const fingerNames = { Thumb: ["Metacarpal", "Proximal", "Distal"], Index: ["Proximal", "Intermediate", "Distal"], Middle: ["Proximal", "Intermediate", "Distal"], Ring: ["Proximal", "Intermediate", "Distal"], Pinky: ["Proximal", "Intermediate", "Distal"] };
  for (const [finger, joints] of Object.entries(fingerNames)) {
    joints.forEach((joint, index) => {
      const vrmFinger = finger === "Pinky" ? "Little" : finger;
      boneMap[`${side}Hand${finger}${index + 1}`] = `${lower}${vrmFinger}${joint}`;
    });
  }
}

function sourceBone(root, rawName) {
  const plain = rawName.replace(/^mixamorig:?/, "");
  return root.getObjectByName(rawName) || root.getObjectByName(`mixamorig${plain}`) || root.getObjectByName(`mixamorig:${plain}`);
}

function retargetClip(fbx, name) {
  const sourceClip = THREE.AnimationClip.findByName(fbx.animations, "mixamo.com") || fbx.animations[0];
  if (!sourceClip) throw new Error(`No animation data in ${name}`);
  fbx.updateMatrixWorld(true);
  vrm.scene.updateMatrixWorld(true);
  const vrmHips = vrm.humanoid.getNormalizedBoneNode("hips");
  const mixamoHips = sourceBone(fbx, "mixamorigHips");
  const vrmHeight = vrmHips.getWorldPosition(new THREE.Vector3()).y - vrm.scene.getWorldPosition(new THREE.Vector3()).y;
  const sourceHeight = Math.abs(mixamoHips?.position.y || 1);
  const hipsScale = vrmHeight / sourceHeight;
  const isVrm0 = vrm.meta?.metaVersion === "0";
  const tracks = [];

  for (const track of sourceClip.tracks) {
    const dot = track.name.lastIndexOf(".");
    if (dot < 0) continue;
    const sourceName = track.name.slice(0, dot);
    const property = track.name.slice(dot + 1);
    const plainName = sourceName.replace(/^mixamorig:?/, "");
    const humanoidName = boneMap[plainName];
    const target = humanoidName && vrm.humanoid.getNormalizedBoneNode(humanoidName);
    const source = sourceBone(fbx, sourceName);
    if (!target || !source || !["quaternion", "position"].includes(property)) continue;

    const values = track.values.slice();
    if (property === "quaternion") {
      const restInverse = source.getWorldQuaternion(new THREE.Quaternion()).invert();
      const parentRest = source.parent?.getWorldQuaternion(new THREE.Quaternion()) || new THREE.Quaternion();
      const value = new THREE.Quaternion();
      for (let i = 0; i < values.length; i += 4) {
        value.fromArray(values, i).premultiply(parentRest).multiply(restInverse);
        if (isVrm0) { value.x *= -1; value.z *= -1; }
        value.toArray(values, i);
      }
      tracks.push(new THREE.QuaternionKeyframeTrack(`${target.name}.quaternion`, track.times, values));
    } else if (humanoidName === "hips") {
      for (let i = 0; i < values.length; i += 3) {
        values[i] *= hipsScale;
        values[i + 1] *= hipsScale;
        values[i + 2] *= hipsScale;
        if (isVrm0) { values[i] *= -1; values[i + 2] *= -1; }
      }
      tracks.push(new THREE.VectorKeyframeTrack(`${target.name}.position`, track.times, values));
    }
  }
  return new THREE.AnimationClip(name, sourceClip.duration, tracks);
}

async function loadCharacter() {
  try {
    const loader = new GLTFLoader();
    loader.register((parser) => new VRMLoaderPlugin(parser));
    const gltf = await loader.loadAsync("/avatars/CoolPineapple.vrm");
    vrm = gltf.userData.vrm;
    VRMUtils.removeUnnecessaryVertices(gltf.scene);
    VRMUtils.combineSkeletons(gltf.scene);
    VRMUtils.rotateVRM0(vrm);
    vrm.scene.traverse((object) => {
      object.frustumCulled = false;
      if (object.isMesh) { object.castShadow = true; object.receiveShadow = true; }
    });
    scene.add(vrm.scene);
    mixer = new THREE.AnimationMixer(vrm.scene);

    const fbxLoader = new FBXLoader();
    const names = ["idle", "thinking", "talking", "dance", "fall-flat", "stand-up"];
    const results = await Promise.allSettled(names.map(async (name) => {
      const fbx = await fbxLoader.loadAsync(`/animations/${name}.fbx`);
      clips.set(name, retargetClip(fbx, name));
    }));
    const loaded = results.filter((result) => result.status === "fulfilled").length;
    if (!clips.has("idle")) throw new Error("Idle animation could not load");
    playAnimation("idle");
    loadState.hidden = true;
    danceButton.disabled = !started || !clips.has("dance");
    modelReadyResolve(true);
    if (loaded !== names.length) updateStatus(`Ready · ${loaded}/${names.length} moves`, true);
  } catch (error) {
    loadState.textContent = "3D character unavailable — chat still works";
    modelReadyResolve(false);
  }
}

function playAnimation(name, { loop = true, fade = 0.4, startAt = 0, speed = 1 } = {}) {
  if (!mixer || !clips.has(name)) return null;
  if (currentAnimation === name && currentAction?.isRunning()) return currentAction;
  const next = mixer.clipAction(clips.get(name));
  next.reset().setEffectiveTimeScale(speed).setEffectiveWeight(1);
  next.setLoop(loop ? THREE.LoopRepeat : THREE.LoopOnce, loop ? Infinity : 1);
  next.clampWhenFinished = !loop;
  next.time = Math.min(startAt, Math.max(0, clips.get(name).duration - .05));
  next.play();
  if (currentAction && currentAction !== next) currentAction.crossFadeTo(next, fade, false);
  currentAction = next;
  currentAnimation = name;
  return next;
}

function playOnce(name, options = {}) {
  return new Promise((resolve) => {
    const action = playAnimation(name, { ...options, loop: false });
    if (!action || !mixer) { resolve(); return; }
    const fallback = setTimeout(done, Math.max(1200, (action.getClip().duration - (options.startAt || 0)) * 1000 / (options.speed || 1) + 500));
    function done(event) {
      if (event?.action && event.action !== action) return;
      clearTimeout(fallback);
      mixer.removeEventListener("finished", done);
      resolve();
    }
    mixer.addEventListener("finished", done);
  });
}

function dropCharacter(landingY, duration = 1050) {
  const startingY = landingY + 4.8;
  const startedAt = performance.now();
  vrm.scene.position.y = startingY;
  return new Promise((resolve) => {
    function frame(now) {
      const progress = Math.min(1, (now - startedAt) / duration);
      const acceleratedFall = progress * progress;
      vrm.scene.position.y = THREE.MathUtils.lerp(startingY, landingY, acceleratedFall);
      if (progress < 1) requestAnimationFrame(frame);
      else {
        vrm.scene.position.y = landingY;
        resolve();
      }
    }
    requestAnimationFrame(frame);
  });
}

async function playIntro() {
  if (!vrm || matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  const landingY = vrm.scene.position.y;
  vrm.scene.position.y = landingY + 4.8;
  await new Promise((resolve) => setTimeout(resolve, 420));
  await Promise.all([
    playOnce("fall-flat", { fade: 0.1 }),
    dropCharacter(landingY),
  ]);
  await playOnce("stand-up", { startAt: 2.5, speed: 1.6 });
  playAnimation("idle");
}

function resize() {
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  renderer.setSize(rect.width, rect.height, false);
  camera.aspect = rect.width / rect.height;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(canvas);

const mouthNames = ["aa", "ih", "ou", "ee", "oh"];
const visemeShapes = {
  a: { aa: 1 }, "@": { aa: .5 }, e: { ee: .8 }, E: { ee: .6, aa: .3 }, i: { ih: .8 },
  o: { oh: .9 }, O: { oh: .7, aa: .3 }, u: { ou: .9 }, t: { ih: .3 }, s: { ih: .3 }, T: { ih: .3 },
  S: { ou: .3, ih: .2 }, k: { aa: .3 }, r: { ou: .3 }, f: { ih: .2 }
};

function setExpression(name, value) {
  try { vrm?.expressionManager?.setValue(name, value); } catch { /* Avatar may not expose every expression. */ }
}
function setEmotion(next) {
  for (const name of ["happy", "angry", "sad", "relaxed"]) setExpression(name, 0);
  emotion = next;
  if (next !== "neutral") setExpression(next, .7);
}
function closeMouth() { mouthNames.forEach((name) => setExpression(name, 0)); }

function updateFace(now) {
  if (!vrm) return;
  if (now >= blinkAt && !blinkStarted) blinkStarted = now;
  if (blinkStarted) {
    const progress = (now - blinkStarted) / 180;
    setExpression("blink", progress < .5 ? progress * 2 : Math.max(0, (1 - progress) * 2));
    if (progress >= 1) { blinkStarted = 0; blinkAt = now + 2600 + Math.random() * 3400; }
  }
  if (lipSync) {
    const latency = lipSync.context.outputLatency || 0;
    const elapsed = (lipSync.context.currentTime - lipSync.startedAt - latency) * 1000;
    let active;
    for (const mark of lipSync.marks) {
      if (mark.time > elapsed) break;
      if (mark.type === "viseme") active = mark.value;
    }
    closeMouth();
    const shape = visemeShapes[active] || {};
    Object.entries(shape).forEach(([name, value]) => setExpression(name, value));
  } else if (fallbackMouth) {
    closeMouth();
    setExpression("aa", .15 + Math.abs(Math.sin(now / 92)) * .48);
  }
}

function animate(now = performance.now()) {
  requestAnimationFrame(animate);
  const delta = Math.min(clock.getDelta(), .05);
  mixer?.update(delta);
  updateFace(now);
  vrm?.update(delta);
  renderer.render(scene, camera);
}
animate();
loadCharacter();

let audioContext;
function ensureAudio() {
  audioContext ||= new AudioContext();
  return audioContext.resume().then(() => audioContext);
}
function decodeBase64(value) {
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes.buffer;
}

async function speak(data) {
  bubble.textContent = data.text;
  bubble.hidden = false;
  setEmotion(data.emotion || "neutral");
  if (!voiceEnabled) {
    await new Promise((resolve) => setTimeout(resolve, Math.min(4500, 1400 + data.text.length * 18)));
    bubble.hidden = true;
    setEmotion("neutral");
    return;
  }
  playAnimation("talking");
  if (data.speech?.audio) {
    try {
      const context = await ensureAudio();
      const audioBuffer = await context.decodeAudioData(decodeBase64(data.speech.audio));
      await new Promise((resolve) => {
        const source = context.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(context.destination);
        lipSync = { context, startedAt: context.currentTime, marks: data.speech.marks || [] };
        source.onended = resolve;
        source.start();
      });
    } catch {
      await browserSpeech(data.text);
    }
  } else {
    await browserSpeech(data.text);
  }
  lipSync = null;
  fallbackMouth = false;
  closeMouth();
  setEmotion("neutral");
  bubble.hidden = true;
  playAnimation("idle");
  if (data.move === "dance") startDance();
}

function browserSpeech(text) {
  return new Promise((resolve) => {
    if (!("speechSynthesis" in window)) { setTimeout(resolve, 1200); return; }
    speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = .96;
    utterance.pitch = 1.04;
    utterance.onstart = () => { fallbackMouth = true; };
    utterance.onend = resolve;
    utterance.onerror = resolve;
    speechSynthesis.speak(utterance);
  });
}

function startDance() {
  if (!clips.has("dance")) return;
  if (currentAnimation === "dance") { stopDance(); return; }
  playAnimation("dance");
  danceAudio ||= new Audio("/music/dance.mp3");
  danceAudio.loop = true;
  danceAudio.currentTime = 0;
  if (voiceEnabled) danceAudio.play().catch(() => {});
  danceButton.innerHTML = '<span aria-hidden="true">■</span> Stop';
}
function stopDance() {
  if (currentAnimation !== "dance") return;
  danceAudio?.pause();
  playAnimation("idle");
  danceButton.innerHTML = '<span aria-hidden="true">♪</span> Dance';
}

danceButton.addEventListener("click", startDance);
voiceToggle.addEventListener("click", () => {
  voiceEnabled = !voiceEnabled;
  voiceToggle.setAttribute("aria-pressed", String(voiceEnabled));
  voiceToggle.setAttribute("aria-label", `Turn voice ${voiceEnabled ? "off" : "on"}`);
  voiceToggle.title = `Voice ${voiceEnabled ? "on" : "off"}`;
  if (!voiceEnabled) { speechSynthesis.cancel(); danceAudio?.pause(); }
});

function updateStatus(label, ready = connected) {
  serviceState.classList.toggle("connected", ready);
  serviceState.querySelector("span:last-child").textContent = label;
}
function serviceLabel(services = {}) {
  const brain = services.bedrock === "available" ? "Nova ready" : services.bedrock === "unavailable" || services.bedrock === "disabled" ? "Local fallback" : "Cloud ready";
  const voice = services.polly === "available" ? "Polly voice" : services.polly === "unavailable" || services.polly === "disabled" ? "Browser voice" : "Voice ready";
  return `${brain} · ${voice}`;
}
function setControls() {
  const enabled = connected && started && !busy;
  messageInput.disabled = !enabled;
  sendButton.disabled = !enabled || !messageInput.value.trim();
  taskForm.querySelector("button").disabled = !enabled || !taskTitle.value.trim();
  summaryButton.disabled = !enabled;
  danceButton.disabled = !started || !clips.has("dance");
  micButton.disabled = !enabled || !microphoneSupported();
}
function setBusy(value) {
  busy = value;
  if (value) { stopDance(); playAnimation("thinking"); }
  setControls();
}

function connect() {
  clearTimeout(reconnectTimer);
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${protocol}://${location.host}/ws`);
  socket.addEventListener("open", () => {
    connected = true;
    socket.send(JSON.stringify({ type: "hello", client_id: clientId, timezone: Intl.DateTimeFormat().resolvedOptions().timeZone }));
    updateStatus("Connected", true);
    setControls();
  });
  socket.addEventListener("close", () => {
    connected = false;
    updateStatus("Reconnecting…", false);
    setControls();
    reconnectTimer = setTimeout(connect, 1800);
  });
  socket.addEventListener("message", async (event) => {
    const data = JSON.parse(event.data);
    if (data.services) {
      serviceStatus = data.services;
      updateStatus(serviceLabel(data.services), true);
      setControls();
    }
    if (data.tasks) { tasks = data.tasks; renderTasks(); }
    if (data.type === "reply") {
      addLog(data.text, "assistant");
      await speak(data);
      setBusy(false);
      messageInput.focus();
    } else if (data.type === "transcript_ready") {
      transcript.textContent = "Listening with Amazon Transcribe…";
    } else if (data.type === "transcript") {
      transcript.textContent = data.text || "Listening…";
      transcript.hidden = false;
    } else if (data.type === "transcript_final") {
      transcript.textContent = data.text;
      addLog(data.text, "user");
    } else if (data.type === "error" || data.type === "transcript_error") {
      cleanupRecording();
      addLog(data.text, "assistant");
      bubble.textContent = data.text;
      bubble.hidden = false;
      setTimeout(() => { bubble.hidden = true; }, 3500);
      setBusy(false);
    } else if (data.type === "confirm") {
      showConfirmation(data);
      setBusy(false);
    }
  });
}
connect();

function send(payload) {
  if (!connected || socket.readyState !== WebSocket.OPEN) return false;
  socket.send(JSON.stringify(payload));
  return true;
}
function submitMessage(text) {
  const cleaned = text.replace(/\s+/g, " ").trim().slice(0, 500);
  if (!cleaned || busy) return;
  addLog(cleaned, "user");
  messageInput.value = "";
  count.textContent = "0/500";
  setBusy(true);
  send({ type: "chat", text: cleaned });
}
composer.addEventListener("submit", (event) => { event.preventDefault(); submitMessage(messageInput.value); });
messageInput.addEventListener("input", () => {
  count.textContent = `${messageInput.value.length}/500`;
  setControls();
});
taskTitle.addEventListener("input", setControls);

function addLog(text, who) {
  const item = document.createElement("div");
  item.className = `log-message ${who}`;
  item.textContent = text;
  conversationLog.append(item);
  while (conversationLog.children.length > 3) conversationLog.firstElementChild.remove();
}

function renderTasks() {
  taskList.replaceChildren();
  if (!tasks.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "Nothing here yet. Add a task, or tell Nour what you need to remember.";
    taskList.append(empty);
    return;
  }
  for (const task of tasks) {
    const fragment = taskTemplate.content.cloneNode(true);
    const item = fragment.querySelector(".task-item");
    const check = fragment.querySelector(".task-check");
    const title = fragment.querySelector("strong");
    const time = fragment.querySelector("time");
    const remove = fragment.querySelector(".task-delete");
    title.textContent = task.title;
    item.classList.toggle("complete", task.completed);
    check.setAttribute("aria-label", task.completed ? `Reopen ${task.title}` : `Complete ${task.title}`);
    if (task.due_at) {
      const due = new Date(task.due_at);
      time.dateTime = task.due_at;
      time.textContent = due.toLocaleString([], { weekday: "short", month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
      item.classList.toggle("overdue", !task.completed && due < new Date());
    } else {
      time.textContent = "No reminder set";
    }
    check.addEventListener("click", () => { setBusy(true); send({ type: "task_toggle", id: task.id, completed: !task.completed }); });
    remove.addEventListener("click", () => { setBusy(true); send({ type: "task_delete", id: task.id }); });
    taskList.append(fragment);
  }
}

taskForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const title = taskTitle.value.trim();
  if (!title || busy) return;
  const due = taskDue.value ? new Date(taskDue.value).toISOString() : null;
  taskTitle.value = "";
  taskDue.value = "";
  setBusy(true);
  send({ type: "task_create", title, due_at: due });
});
summaryButton.addEventListener("click", () => { setBusy(true); send({ type: "summary" }); });

function showConfirmation(data) {
  document.querySelector(".confirm-bar")?.remove();
  const bar = document.createElement("div");
  bar.className = "confirm-bar";
  bar.setAttribute("role", "alertdialog");
  const text = document.createElement("span");
  text.textContent = data.text;
  const cancel = document.createElement("button");
  cancel.type = "button";
  cancel.className = "cancel";
  cancel.textContent = "Keep";
  const confirm = document.createElement("button");
  confirm.type = "button";
  confirm.textContent = "Delete";
  cancel.addEventListener("click", () => bar.remove());
  confirm.addEventListener("click", () => { bar.remove(); setBusy(true); send({ type: "confirm", token: data.token }); });
  bar.append(text, cancel, confirm);
  document.body.append(bar);
  confirm.focus();
}

startButton.addEventListener("click", async () => {
  startButton.disabled = true;
  await ensureAudio().catch(() => {});
  started = true;
  startScreen.classList.add("closed");
  setControls();
  await modelReady;
  await playIntro();
  danceButton.disabled = !clips.has("dance");
  if (connected) send({ type: "greeting" });
});

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const secureMic = location.hostname === "localhost" || location.hostname === "127.0.0.1" || location.protocol === "https:";
const recognitionSupported = Boolean(SpeechRecognition && secureMic);
let recognition;
let recording = false;
let recordingMode = null;
let speechFinal = "";
let recordingTimer;
let microphoneStream;
let captureContext;
let captureSource;
let captureWorklet;
let silentGain;

function microphoneSupported() {
  const awsAvailable = serviceStatus.transcribe !== "unavailable" && serviceStatus.transcribe !== "disabled";
  return secureMic && Boolean(navigator.mediaDevices?.getUserMedia) && (awsAvailable || recognitionSupported);
}

if (!secureMic) micButton.hidden = true;
if (recognitionSupported) {
  recognition = new SpeechRecognition();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.lang = "en-US";
  recognition.onresult = (event) => {
    let interim = "";
    for (let i = event.resultIndex; i < event.results.length; i += 1) {
      const words = event.results[i][0].transcript;
      if (event.results[i].isFinal) speechFinal += `${words} `;
      else interim += words;
    }
    transcript.textContent = (speechFinal + interim).trim() || "Listening…";
  };
  recognition.onerror = (event) => {
    const message = event.error === "not-allowed" ? "Microphone permission was not granted." : "I couldn’t hear that. Try again.";
    stopRecording(false);
    transcript.textContent = message;
    transcript.hidden = false;
    setTimeout(() => { transcript.hidden = true; }, 2200);
  };
  recognition.onend = () => {
    if (recording && recordingMode === "browser") stopRecording(true);
  };
}

function updateRecordingUi(active) {
  micButton.classList.toggle("recording", active);
  micButton.querySelector("span:last-child").textContent = active ? "Listening…" : "Hold to talk";
  setControls();
}

function cleanupRecording() {
  clearTimeout(recordingTimer);
  captureSource?.disconnect();
  captureWorklet?.disconnect();
  silentGain?.disconnect();
  microphoneStream?.getTracks().forEach((track) => track.stop());
  if (captureContext && captureContext.state !== "closed") captureContext.close().catch(() => {});
  captureSource = null;
  captureWorklet = null;
  silentGain = null;
  microphoneStream = null;
  captureContext = null;
  recording = false;
  recordingMode = null;
  updateRecordingUi(false);
}

async function startAwsRecording() {
  microphoneStream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  });
  if (!recording) {
    microphoneStream.getTracks().forEach((track) => track.stop());
    microphoneStream = null;
    return;
  }
  captureContext = new AudioContext();
  await captureContext.audioWorklet.addModule("/pcm-worklet.js");
  await captureContext.resume();
  if (!recording) {
    cleanupRecording();
    return;
  }
  captureSource = captureContext.createMediaStreamSource(microphoneStream);
  captureWorklet = new AudioWorkletNode(captureContext, "pcm-16k-processor");
  silentGain = captureContext.createGain();
  silentGain.gain.value = 0;
  captureWorklet.port.onmessage = (event) => {
    if (recording && recordingMode === "aws" && socket?.readyState === WebSocket.OPEN) {
      socket.send(event.data);
    }
  };
  if (!send({ type: "transcribe_start" })) throw new Error("WebSocket is not connected");
  recordingMode = "aws";
  captureSource.connect(captureWorklet);
  captureWorklet.connect(silentGain);
  silentGain.connect(captureContext.destination);
}

function startBrowserRecording() {
  if (!recognitionSupported) throw new Error("Browser speech recognition is unavailable");
  recordingMode = "browser";
  speechFinal = "";
  recognition.start();
}

async function startRecording() {
  if (!microphoneSupported() || recording || busy) return;
  recording = true;
  speechFinal = "";
  transcript.textContent = "Starting microphone…";
  transcript.hidden = false;
  updateRecordingUi(true);
  const preferAws = serviceStatus.transcribe !== "unavailable" && serviceStatus.transcribe !== "disabled";
  try {
    if (preferAws) await startAwsRecording();
    else startBrowserRecording();
    if (recording) recordingTimer = setTimeout(() => stopRecording(true), 30000);
  } catch {
    cleanupRecording();
    if (recognitionSupported) {
      recording = true;
      updateRecordingUi(true);
      try {
        startBrowserRecording();
        recordingTimer = setTimeout(() => stopRecording(true), 30000);
        return;
      } catch { /* Fall through to the visible error. */ }
    }
    transcript.textContent = "The microphone could not start. Check browser permission and retry.";
    transcript.hidden = false;
    setTimeout(() => { transcript.hidden = true; }, 2500);
  }
}

function stopRecording(submit = true) {
  if (!recording) return;
  const mode = recordingMode;
  recording = false;
  clearTimeout(recordingTimer);
  updateRecordingUi(false);
  if (mode === "browser") {
    try { recognition.stop(); } catch { /* Recognition may already be stopped. */ }
    const words = speechFinal.trim();
    recordingMode = null;
    setTimeout(() => { transcript.hidden = true; }, words ? 400 : 1800);
    if (submit && words) submitMessage(words);
    return;
  }
  if (mode === "aws") {
    captureSource?.disconnect();
    captureWorklet?.disconnect();
    silentGain?.disconnect();
    microphoneStream?.getTracks().forEach((track) => track.stop());
    if (captureContext && captureContext.state !== "closed") captureContext.close().catch(() => {});
    captureSource = null;
    captureWorklet = null;
    silentGain = null;
    microphoneStream = null;
    captureContext = null;
    recordingMode = null;
    transcript.textContent = "Understanding your message…";
    if (submit) {
      setBusy(true);
      send({ type: "transcribe_stop" });
    }
    return;
  }
  cleanupRecording();
}

micButton.addEventListener("pointerdown", (event) => { event.preventDefault(); startRecording(); });
window.addEventListener("pointerup", () => stopRecording(true));
window.addEventListener("keydown", (event) => {
  const typing = ["INPUT", "TEXTAREA"].includes(document.activeElement?.tagName);
  if (event.code === "Space" && !typing && !event.repeat) { event.preventDefault(); startRecording(); }
});
window.addEventListener("keyup", (event) => {
  const typing = ["INPUT", "TEXTAREA"].includes(document.activeElement?.tagName);
  if (event.code === "Space" && !typing) { event.preventDefault(); stopRecording(true); }
});
setControls();
