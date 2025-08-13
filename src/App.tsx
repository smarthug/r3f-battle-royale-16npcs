import React, { useMemo, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Html } from "@react-three/drei";
import type { Mesh } from "three";
import {
  Physics,
  RigidBody,
  CapsuleCollider,
  CuboidCollider,
} from "@react-three/rapier";

// --- Tunables ---------------------------------------------------------------
const ARENA_RADIUS = 32; // world units
const BOT_COUNT = 16;
const BOT_SPEED = 5.2; // u/s
const BOT_ACCEL = 14; // u/s^2
const BOT_SEPARATION = 2.2; // desired min distance
const ATTACK_RANGE = 1.6;
const ATTACK_DPS = 22; // damage per second
const ATTACK_COOLDOWN = 0.45; // seconds between hits
const FOV = Math.PI * 1.1; // ~200 degrees
const ZONE_START_RADIUS = ARENA_RADIUS * 0.95;
const ZONE_END_RADIUS = 4;
const GAME_DURATION = 180; // seconds end-to-end shrink time
const STORM_DPS = 18; // damage when outside the zone

// --- Types -----------------------------------------------------------------
type Vec3 = [number, number, number];
interface BotState {
  id: number;
  hp: number; // 0..100
  alive: boolean;
  lastHitAt: number; // time of last successful attack
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
  const [bots, setBots] = useState<BotState[]>(() => {
    const arr: BotState[] = [];
    for (let i = 0; i < BOT_COUNT; i++) {
      arr.push({ id: i, hp: 100, alive: true, lastHitAt: -999, targetId: null });
    }
    return arr;
  });

  const zone = useRef({ t: 0, radius: ZONE_START_RADIUS });
  const aliveCount = useMemo(() => bots.filter((b) => b.alive).length, [bots]);
  const winner = aliveCount === 1 ? bots.find((b) => b.alive) ?? null : null;

  useFrame((_, dt) => {
    const now = performance.now() / 1000;
    const next = bots.map((b) => ({ ...b }));

    // Zone shrink
    zone.current.t = clamp(zone.current.t + dt, 0, GAME_DURATION);
    const k = zone.current.t / GAME_DURATION;
    zone.current.radius = ZONE_START_RADIUS * (1 - k) + ZONE_END_RADIUS * k;

    const getPos = (id: number) => {
  const rb = bodies.current[id];
      const p = rb?.translation();
      return p ? ([p.x, p.y, p.z] as Vec3) : ([0, 0, 0] as Vec3);
    };
    const getVel = (id: number) => {
  const rb = bodies.current[id];
      const v = rb?.linvel();
      return v ? ([v.x, 0, v.z] as Vec3) : ([0, 0, 0] as Vec3);
    };

    for (const me of next) {
      if (!me.alive) continue;
      const myPos = getPos(me.id);
      const myVel = getVel(me.id);

      // Storm damage if outside zone
      const dFromCenter = len2(myPos[0], 0, myPos[2]);
      if (dFromCenter > zone.current.radius) {
        me.hp -= STORM_DPS * dt;
      }

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

      // Keep roughly inside arena bounds (soft bias, walls handle hard limit)
      const radial = len2(myPos[0], 0, myPos[2]);
      if (radial > ARENA_RADIUS * 0.9) {
        const pull = mul(norm(-myPos[0], 0, -myPos[2]), 2.5);
        desired = add(desired, pull);
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
  bodies.current[me.id]?.setLinvel({ x: newVx, y: 0, z: newVz }, true);

      // Attempt attack if in range & cooldown
      if (target) {
        const tPos = getPos(target.id);
        const to = sub(tPos, myPos);
        const dist = len2(to[0], to[1], to[2]);
        if (dist <= ATTACK_RANGE && now - me.lastHitAt >= ATTACK_COOLDOWN) {
          const tgt = next[target.id];
          if (tgt) tgt.hp -= ATTACK_DPS * ATTACK_COOLDOWN;
          me.lastHitAt = now;
        }
      }

      // Death handling
      if (me.hp <= 0) {
        me.alive = false;
        // Disable physics body so it no longer interacts
  bodies.current[me.id]?.setEnabled(false);
      }
    }

    setBots(next);
  });

  return { bots, zone, aliveCount, winner } as const;
}

// --- Renderables ------------------------------------------------------------
function Arena({ radius }: { radius: number }) {
  return (
    <group>
      {/* floor visual + physics ground */}
      <RigidBody type="fixed" colliders={false}>
        <mesh rotation-x={-Math.PI / 2} receiveShadow>
          <circleGeometry args={[radius, 64]} />
          <meshStandardMaterial roughness={1} metalness={0} color="#2a2a2a" />
        </mesh>
        {/* simple large ground box as collider */}
        <CuboidCollider args={[radius, 0.05, radius]} position={[0, -0.05, 0]} />
      </RigidBody>

      {/* wall ring (visual) */}
      <mesh rotation-x={-Math.PI / 2} position={[0, 0.01, 0]}>
        <ringGeometry args={[radius * 0.98, radius, 64]} />
        <meshBasicMaterial color="#3a3a3a" />
      </mesh>

      {/* Physical boundary walls approximated by small boxes in a ring */}
      {Array.from({ length: 48 }).map((_, i) => {
        const a = (i / 48) * Math.PI * 2;
        const x = Math.sin(a) * (radius - 0.5);
        const z = Math.cos(a) * (radius - 0.5);
        const rot = Math.atan2(Math.sin(a), Math.cos(a));
        return (
          <RigidBody key={i} type="fixed" colliders={false} position={[x, 0.75, z]} rotation={[0, rot, 0]}>
            <CuboidCollider args={[0.5, 0.75, 2]} />
          </RigidBody>
        );
      })}
    </group>
  );
}

function Zone({
  zoneRef,
}: {
  zoneRef: React.MutableRefObject<{ t: number; radius: number }>;
}) {
  const [r, setR] = useState(zoneRef.current.radius);
  useFrame(() => setR(zoneRef.current.radius));
  return (
    <group>
      {/* safe circle */}
      <mesh rotation-x={-Math.PI / 2} position={[0, 0.02, 0]} renderOrder={2}>
        <ringGeometry args={[Math.max(0, r - 0.2), r, 64]} />
        <meshBasicMaterial transparent opacity={0.9} />
      </mesh>
      {/* fill outside (storm) */}
      <mesh rotation-x={-Math.PI / 2} position={[0, 0.01, 0]} renderOrder={1}>
        <ringGeometry args={[r, ARENA_RADIUS * 1.01, 64]} />
        <meshBasicMaterial color="#7e0b0b" transparent opacity={0.18} />
      </mesh>
    </group>
  );
}

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
      enabledRotations={[false, true, false]}
      canSleep={false}
    >
      {/* physical collider */}
      <CapsuleCollider args={[0.45, 0.4]} />
      {/* visual */}
      <mesh ref={meshRef} castShadow position={[0, 0.5, 0]}>
        <capsuleGeometry args={[0.4, 0.9, 6, 12]} />
        <meshStandardMaterial color={color} />
      </mesh>
      {/* HP bar (follows body) */}
      {bot.alive && (
        <Html position={[0, 1.6, 0]} center distanceFactor={12} style={{ pointerEvents: "none" }}>
          <div style={{ width: 48, height: 6, background: "#222", borderRadius: 4 }}>
            <div
              style={{
                width: `${clamp(bot.hp, 0, 100)}%`,
                height: "100%",
                background: "#38b000",
                borderRadius: 4,
              }}
            />
          </div>
        </Html>
      )}
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
      <Canvas shadows camera={{ position: [0, 24, 30], fov: 50 }}>
        <color attach="background" args={["#0b0b0b"]} />
        <ambientLight intensity={0.35} />
        <directionalLight
          position={[10, 18, 12]}
          intensity={1}
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

  const { bots, zone, aliveCount, winner } = useBattleRoyalePhysics(bodyRefs);

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
        <Zone zoneRef={zone} />
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
