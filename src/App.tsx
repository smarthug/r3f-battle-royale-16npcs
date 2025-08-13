import React, { useMemo, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Html, Sky } from "@react-three/drei";
import type { Mesh } from "three";
import { Physics, RigidBody, CapsuleCollider, CuboidCollider } from "@react-three/rapier";

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

  // No keep-in-zone bias: allow ring-outs

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

  // No height-based elimination in wall-touch mode
    }

  setBots(next);
  });

  return { bots, aliveCount, winner, eliminate } as const;
}

// --- Renderables ------------------------------------------------------------
function Arena({ radius }: { radius: number }) {
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

      {/* wall ring (visual) */}
      <mesh rotation-x={-Math.PI / 2} position={[0, 0.01, 0]}>
        <ringGeometry args={[radius * 0.98, radius, 64]} />
        <meshBasicMaterial color="#dfeeea" />
      </mesh>

      {/* Physical boundary walls in a ring (elimination on touch) */}
      {Array.from({ length: 48 }).map((_, i) => {
        const a = (i / 48) * Math.PI * 2;
        const x = Math.sin(a) * (radius - 0.25);
        const z = Math.cos(a) * (radius - 0.25);
        const rotY = Math.atan2(Math.sin(a), Math.cos(a));
        return (
          <RigidBody
            key={i}
            type="fixed"
            colliders={false}
            position={[x, 0.75, z]}
            rotation={[0, rotY, 0]}
            name="wall"
            // Mark as wall for collision detection
            userData={{ isWall: true }}
          >
            <CuboidCollider args={[0.25, 0.75, 1.2]} restitution={0} friction={0.8} />
          </RigidBody>
        );
      })}
    </group>
  );
}

// Zone removed for ring-out mode

function Bot({ bot, setBody, onHitWall }: { bot: BotState; setBody: (api: any | null) => void; onHitWall: (id: number) => void }) {
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
      onCollisionEnter={(e: any) => {
        const otherObj = e?.other?.rigidBodyObject || e?.other?.colliderObject;
        const isWall = otherObj?.userData?.isWall || otherObj?.name === "wall";
        if (isWall && bot.alive) onHitWall(bot.id);
      }}
    >
      {/* physical collider */}
      <CapsuleCollider args={[0.45, 0.4]} />
      {/* visual */}
      <mesh ref={meshRef} castShadow position={[0, 0.5, 0]}>
        <capsuleGeometry args={[0.4, 0.9, 6, 12]} />
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

  const { bots, aliveCount, winner, eliminate } = useBattleRoyalePhysics(bodyRefs);

  // After bodies mount, set spawn positions (once)
  useFrame(() => {
    if (initialized.current) return;
    if (bodyRefs.current.length < BOT_COUNT) return;
    for (let i = 0; i < BOT_COUNT; i++) {
      const r = rand(6, ARENA_RADIUS * 0.8);
      const t = rand(0, Math.PI * 2);
      const x = Math.sin(t) * r;
      const z = Math.cos(t) * r;
      bodyRefs.current[i]?.setTranslation({ x, y: 0.5, z }, false);
      bodyRefs.current[i]?.setLinvel({ x: 0, y: 0, z: 0 }, false);
    }
    initialized.current = true;
  });

  return (
    <>
      <group position={[0, 0, 0]}>
  <Arena radius={ARENA_RADIUS} />
    {bots.map((b) => (
          <Bot
            key={b.id}
            bot={b}
            setBody={(api) => {
              if (api) bodyRefs.current[b.id] = api;
              else delete bodyRefs.current[b.id];
            }}
      onHitWall={(id) => eliminate(id)}
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
