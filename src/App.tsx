import React, { useMemo, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Html, Sky } from "@react-three/drei";
import type { Mesh, MeshBasicMaterial } from "three";
import { Physics, RigidBody, CapsuleCollider } from "@react-three/rapier";

// --- Tunables ---------------------------------------------------------------
const ARENA_RADIUS = 16; // world units (smaller map)
const BOT_COUNT = 16;
const BOT_SPEED = 6.0; // u/s
const BOT_ACCEL = 16; // u/s^2
const BOT_SEPARATION = 2.0; // desired min distance
const FOV = Math.PI * 1.1; // ~200 degrees
const PUSH_RANGE = 1.3; // distance to trigger push
const PUSH_COOLDOWN = 0.4; // seconds between pushes
const PUSH_IMPULSE = 5.5; // impulse magnitude for push
// Shrinking safe zone
const SAFE_MIN_RADIUS = 3.5; // how small the stage can get
const SAFE_SHRINK_DURATION = 120; // seconds from full to min
const SAFE_SHRINK_RATE = (ARENA_RADIUS - SAFE_MIN_RADIUS) / SAFE_SHRINK_DURATION; // units per second
// Bot capsule dimensions (shared by collider and visual)
const CAPSULE_HALF = 0.45; // half-height of the cylindrical part
const CAPSULE_RADIUS = 0.4; // radius of capsule caps

// --- Types -----------------------------------------------------------------
type Vec3 = [number, number, number];
interface BotState {
  id: number;
  alive: boolean;
  lastPushAt: number; // time of last push
  targetId: number | null;
}

// Utility functions ---------------------------------------------------------
const rand = (min: number, max: number) => min + Math.random() * (max - min);
const clamp = (x: number, a: number, b: number) => Math.min(Math.max(x, a), b);
const len2 = (x: number, y: number, z: number) =>
  Math.sqrt(x * x + y * y + z * z);
const norm = (x: number, y: number, z: number): Vec3 => {
  const l = len2(x, y, z) || 1;
  return [x / l, y / l, z / l];
};
const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul = (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

function headingFromVelXZ(x: number, z: number) {
  const l = Math.hypot(x, z) || 1;
  return Math.atan2(x / l, z / l); // yaw around Y
}

// --- Core Simulation --------------------------------------------------------
function useBattleRoyalePhysics(bodies: React.MutableRefObject<any[]>) {
  // Queue eliminations to avoid setState during the physics tick (useFrame)
  const pendingElimsRef = useRef<Set<number>>(new Set());
  // Shrinking safe zone state
  const [safeRadius, setSafeRadius] = useState<number>(ARENA_RADIUS);
  const safeRadiusRef = useRef<number>(ARENA_RADIUS);
  const [bots, setBots] = useState<BotState[]>(() => {
    const arr: BotState[] = [];
    for (let i = 0; i < BOT_COUNT; i++) {
    arr.push({ id: i, alive: true, lastPushAt: -999, targetId: null });
    }
    return arr;
  });

  const aliveCount = useMemo(() => bots.filter((b) => b.alive).length, [bots]);
  const winner = aliveCount === 1 ? bots.find((b) => b.alive) ?? null : null;

  const eliminate = (id: number) => {
    pendingElimsRef.current.add(id);
  };

  useFrame((_, dt) => {
    const now = performance.now() / 1000;
    const next = bots.map((b) => ({ ...b }));

    // Shrink safe zone over time
    const newSafe = Math.max(
      SAFE_MIN_RADIUS,
      safeRadiusRef.current - SAFE_SHRINK_RATE * Math.max(0, dt)
    );
    safeRadiusRef.current = newSafe;
    setSafeRadius(newSafe);

    // Apply any pending eliminations inside the frame coherently
    if (pendingElimsRef.current.size > 0) {
      for (const id of pendingElimsRef.current) {
        const b = next[id];
        if (b && b.alive) {
          b.alive = false;
          // Disable its rigidbody so it no longer participates in physics
          bodies.current[id]?.setEnabled(false);
        }
      }
      pendingElimsRef.current.clear();
    }

  const getPos = (id: number) => {
  const rb = bodies.current[id];
      const p = rb?.translation();
      return p ? ([p.x, p.y, p.z] as Vec3) : ([0, 0, 0] as Vec3);
    };
    const getVel = (id: number) => {
      const rb = bodies.current[id];
      const v = rb?.linvel();
      return v ? ([v.x, v.y, v.z] as Vec3) : ([0, 0, 0] as Vec3);
    };

    for (const me of next) {
      if (!me.alive) continue;
      const myPos = getPos(me.id);
      const myVel = getVel(me.id);

      // Acquire/validate target
      let target: BotState | null = null;
      if (me.targetId != null) {
        const t = next[me.targetId];
        if (t && t.alive) target = t;
        else me.targetId = null;
      }
      if (!target) {
        let best: BotState | null = null;
        let bestDist = Infinity;
        for (const other of next) {
          if (other.id === me.id || !other.alive) continue;
          const oPos = getPos(other.id);
          const to = sub(oPos, myPos);
          const dist = len2(to[0], to[1], to[2]);
          const forward: Vec3 =
            myVel[0] === 0 && myVel[2] === 0
              ? [0, 0, 1]
              : norm(myVel[0], 0, myVel[2]);
          const dir = norm(to[0], 0, to[2]);
          const facing = Math.acos(clamp(dot(forward, dir), -1, 1));
          if (facing <= FOV * 0.5 && dist < bestDist) {
            best = other;
            bestDist = dist;
          }
        }
        if (best) {
          me.targetId = best.id;
          target = best;
        }
      }

  // Steering: seek target (if any) + separation + keep-in-zone bias
      let desired: Vec3 = [0, 0, 0];
      if (target) {
        const tPos = getPos(target.id);
        desired = sub(tPos, myPos);
      } else {
        desired = mul(norm(-myPos[0], 0, -myPos[2]), 1);
      }

      // Separation
      let sepX = 0,
        sepZ = 0,
        neighbors = 0;
      for (const other of next) {
        if (other.id === me.id || !other.alive) continue;
        const oPos = getPos(other.id);
        const dx = myPos[0] - oPos[0];
        const dz = myPos[2] - oPos[2];
        const d = Math.hypot(dx, dz);
        if (d > 0 && d < BOT_SEPARATION) {
          const f = (BOT_SEPARATION - d) / BOT_SEPARATION;
          sepX += (dx / d) * f;
          sepZ += (dz / d) * f;
          neighbors++;
        }
      }
      if (neighbors > 0) {
        desired = add(desired, [sepX, 0, sepZ]);
      }

      // Keep-in-zone bias when nearing the shrinking boundary
      const r = Math.hypot(myPos[0], myPos[2]);
      const threshold = safeRadiusRef.current * 0.9;
      if (r > threshold) {
        const inward = norm(-myPos[0], 0, -myPos[2]);
        // Scale bias by how far we are past the threshold
        const bias = clamp((r - threshold) / Math.max(0.001, safeRadiusRef.current - threshold), 0, 1);
        desired = add(desired, mul(inward as Vec3, 2.0 * bias));
      }

      // Acceleration toward desired, then set linvel on rigidbody
      const desiredDir = norm(desired[0], 0, desired[2]);
      const targetVel = mul(desiredDir, BOT_SPEED);
      const dv = sub(targetVel, myVel);
      const maxDv = BOT_ACCEL * dt;
      const dvLen = Math.hypot(dv[0], dv[2]);
      const apply = dvLen > maxDv ? mul(dv, maxDv / (dvLen || 1)) : dv;
  const newVx = myVel[0] + apply[0];
  const newVz = myVel[2] + apply[2];
  // Preserve current vertical velocity so gravity can act and allow falling
  bodies.current[me.id]?.setLinvel({ x: newVx, y: myVel[1], z: newVz }, true);

      // Push mechanic: if close to target and off cooldown, apply outward impulse
      if (target) {
        const tPos = getPos(target.id);
        const to = sub(tPos, myPos);
        const dist = len2(to[0], to[1], to[2]);
        if (dist <= PUSH_RANGE && now - me.lastPushAt >= PUSH_COOLDOWN) {
          const dir = norm(to[0], 0, to[2]);
          bodies.current[target.id]?.applyImpulse(
            { x: dir[0] * PUSH_IMPULSE, y: 0, z: dir[2] * PUSH_IMPULSE },
            true
          );
          me.lastPushAt = now;
        }
      }

      // Eliminate if outside the shrinking safe zone
      if (Math.hypot(myPos[0], myPos[2]) > safeRadiusRef.current + 0.02) {
        pendingElimsRef.current.add(me.id);
      }
    }

    setBots(next);
  });

  return { bots, aliveCount, winner, eliminate, safeRadius } as const;
}

// --- Renderables ------------------------------------------------------------
function Arena({ radius, safeRadius }: { radius: number; safeRadius: number }) {
  // Pulsing ring materials
  const mainRingMat = useRef<MeshBasicMaterial>(null!);
  const pulseRingMat = useRef<MeshBasicMaterial>(null!);
  const pulseRingMesh = useRef<Mesh>(null!);

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime();
    const pulse = (Math.sin(t * 2.0) + 1) * 0.5; // 0..1
    if (mainRingMat.current) {
      mainRingMat.current.opacity = 0.65 + 0.25 * pulse;
    }
    if (pulseRingMat.current) {
      pulseRingMat.current.opacity = 0.1 + 0.25 * (1 - pulse);
    }
    if (pulseRingMesh.current) {
      const s = 1 + 0.02 * pulse;
      pulseRingMesh.current.scale.set(s, 1, s);
    }
  });

  return (
    <group>
      {/* floor visual + physics ground */}
      <RigidBody type="fixed" colliders="trimesh">
        <mesh rotation-x={-Math.PI / 2} receiveShadow>
          <circleGeometry args={[radius, 64]} />
          <meshStandardMaterial roughness={1} metalness={0} color="#2ecc71" />
        </mesh>
      </RigidBody>

      {/* island thickness (visual) */}
      {/* <mesh position={[0, -0.6, 0]} receiveShadow castShadow>
        <cylinderGeometry args={[radius * 0.98, radius * 0.98, 1.2, 48]} />
        <meshStandardMaterial color="#7a5a3a" roughness={1} metalness={0} />
      </mesh> */}

      {/* shrinking safe zone (visual rings with subtle pulse) */}
      <mesh rotation-x={-Math.PI / 2} position={[0, 0.012, 0]}>
        <ringGeometry args={[safeRadius * 0.985, safeRadius, 128]} />
        <meshBasicMaterial ref={mainRingMat} color="#ffffff" transparent depthWrite={false} opacity={0.85} />
      </mesh>
      <mesh rotation-x={-Math.PI / 2} position={[0, 0.011, 0]} ref={pulseRingMesh}>
        <ringGeometry args={[safeRadius * 0.99, safeRadius * 1.03, 128]} />
        <meshBasicMaterial ref={pulseRingMat} color="#b6fff3" transparent depthWrite={false} opacity={0.2} />
      </mesh>
    </group>
  );
}

// Zone removed for ring-out mode

function Bot({ bot, setBody }: { bot: BotState; setBody: (api: any | null) => void }) {
  const meshRef = useRef<Mesh>(null!);
  useFrame(() => {
    // We can't read the body directly here; instead, store it on the mesh for quick access
  const rb = (meshRef.current as any)?._rb as any | undefined;
    if (!rb || !meshRef.current) return;
    const v = rb.linvel();
    const yaw = headingFromVelXZ(v.x, v.z);
    meshRef.current.rotation.y = yaw;
    const s = bot.alive ? 1 : 0.2;
    meshRef.current.scale.setScalar(s);
  });

  const color = bot.alive ? `hsl(${(bot.id * 137) % 360}deg 80% 60%)` : "#333";
  return (
    <RigidBody
  ref={(api: any | null) => {
        // expose on mesh for quick reads in frame and pass up to parent
        if (meshRef.current) (meshRef.current as any)._rb = api ?? undefined;
        setBody(api);
      }}
      type="dynamic"
      colliders={false}
      linearDamping={2.2}
      angularDamping={10}
  enabledRotations={[false, false, false]}
  canSleep={false}
    >
      {/* physical collider */}
      <CapsuleCollider args={[CAPSULE_HALF, CAPSULE_RADIUS]} />
      {/* visual (centered at body origin) */}
      <mesh ref={meshRef} castShadow>
        <capsuleGeometry args={[CAPSULE_RADIUS, CAPSULE_HALF * 2, 6, 12]} />
        <meshStandardMaterial color={color} />
      </mesh>
  {/* No HP bar in ring-out mode */}
    </RigidBody>
  );
}

function HUD({ alive, winner }: { alive: number; winner: BotState | null }) {
  return (
    <Html position={[0, 0, 0]} center transform={false}>
      <div
        style={{
          position: "fixed",
          top: 16,
          left: 16,
          padding: 12,
          background: "rgba(0,0,0,0.55)",
          borderRadius: 12,
          color: "white",
          fontFamily: "ui-sans-serif, system-ui",
          backdropFilter: "blur(6px)",
        }}
      >
        <div style={{ fontWeight: 700, marginBottom: 6 }}>
          NPC Battle Royale
        </div>
        <div>Alive: {alive}</div>
        {winner && <div style={{ marginTop: 6 }}>Winner: Bot #{winner.id}</div>}
      </div>
    </Html>
  );
}

// --- Main Component ---------------------------------------------------------
export default function BattleRoyaleR3F() {
  return (
    <div style={{ width: "100%", height: "100%" }}>
      <Canvas shadows camera={{ position: [0, 16, 22], fov: 50 }}>
        <Sky turbidity={6} rayleigh={2.2} mieCoefficient={0.005} mieDirectionalG={0.8} sunPosition={[10, 25, -10]} />
        <ambientLight intensity={0.6} />
        <hemisphereLight color="#ffffff" groundColor="#c7d7ff" intensity={0.8} />
        <directionalLight
          position={[10, 18, 12]}
          intensity={1.4}
          castShadow
          shadow-mapSize-width={2048}
          shadow-mapSize-height={2048}
        />

        <Physics gravity={[0, -9.81, 0]}>
          <TheWorld />
        </Physics>

        <OrbitControls
          enablePan={false}
          minDistance={10}
          maxDistance={90}
          maxPolarAngle={Math.PI * 0.49}
        />
      </Canvas>
    </div>
  );
}

function TheWorld() {
  // Prepare rigid body refs for bots (index by id)
  const bodyRefs = useRef<any[]>([]);

  // Initialize bot spawn positions on first render by placing bodies
  const initialized = useRef(false);

  const { bots, aliveCount, winner, eliminate, safeRadius } = useBattleRoyalePhysics(bodyRefs);

  // After bodies mount, set spawn positions (once)
  useFrame(() => {
    if (initialized.current) return;
    if (bodyRefs.current.length < BOT_COUNT) return;
    for (let i = 0; i < BOT_COUNT; i++) {
      const r = rand(6, ARENA_RADIUS * 0.8);
      const t = rand(0, Math.PI * 2);
      const x = Math.sin(t) * r;
      const z = Math.cos(t) * r;
  // Place body so the capsule just touches the ground (center at half+radius)
  bodyRefs.current[i]?.setTranslation({ x, y: CAPSULE_HALF + CAPSULE_RADIUS + 0.005, z }, false);
      bodyRefs.current[i]?.setLinvel({ x: 0, y: 0, z: 0 }, false);
    }
    initialized.current = true;
  });

  return (
    <>
      <group position={[0, 0, 0]}>
  <Arena radius={ARENA_RADIUS} safeRadius={safeRadius} />
    {bots.map((b) => (
          <Bot
            key={b.id}
            bot={b}
            setBody={(api) => {
              if (api) bodyRefs.current[b.id] = api;
              else delete bodyRefs.current[b.id];
            }}
          />
        ))}
      </group>
      <HUD alive={aliveCount} winner={winner} />
    </>
  );
}

// --- Notes -----------------------------------------------------------------
// • Simplified, single-file demo: steering (seek + separation), shrinking safe-zone,
//   proximity-based melee damage with cooldown, storm damage, and auto-winner detect.
// • To extend: swap steering for NavMesh, add weapons/abilities, raycast LOS, cover system,
//   and behavior tree/utility AI.
// • For production: split files, move sim into a store/ECS, use fixed dt for determinism.
