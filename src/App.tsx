import React, { useMemo, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Html } from "@react-three/drei";

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
  pos: Vec3;
  vel: Vec3;
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

function headingFromVel(v: Vec3) {
  const [x, , z] = v;
  const l = Math.hypot(x, z) || 1;
  return Math.atan2(x / l, z / l); // yaw around Y
}

// --- Core Simulation --------------------------------------------------------
function useBattleRoyale() {
  const [bots, setBots] = useState<BotState[]>(() => {
    const arr: BotState[] = [];
    for (let i = 0; i < BOT_COUNT; i++) {
      // random spawn within a donut to avoid stacking in the center
      const r = rand(6, ARENA_RADIUS * 0.8);
      const t = rand(0, Math.PI * 2);
      arr.push({
        id: i,
        pos: [Math.sin(t) * r, 0, Math.cos(t) * r],
        vel: [0, 0, 0],
        hp: 100,
        alive: true,
        lastHitAt: -999,
        targetId: null,
      });
    }
    return arr;
  });

  const zone = useRef({ t: 0, radius: ZONE_START_RADIUS });
  const aliveCount = useMemo(() => bots.filter((b) => b.alive).length, [bots]);
  const winner = aliveCount === 1 ? bots.find((b) => b.alive) ?? null : null;

  useFrame((_, dt) => {
    const now = performance.now() / 1000;
    const next: BotState[] = bots.map((b) => ({ ...b }));

    // Zone shrink
    zone.current.t = clamp(zone.current.t + dt, 0, GAME_DURATION);
    const k = zone.current.t / GAME_DURATION;
    zone.current.radius = ZONE_START_RADIUS * (1 - k) + ZONE_END_RADIUS * k;

    // Precompute positions for quick reads
    // const positions = next.map((b) => b.pos);

    for (const me of next) {
      if (!me.alive) continue;

      // Storm damage if outside zone
      const dFromCenter = len2(me.pos[0], 0, me.pos[2]);
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
        // pick nearest visible
        let best: BotState | null = null;
        let bestDist = Infinity;
        for (const other of next) {
          if (other.id === me.id || !other.alive) continue;
          const to = sub(other.pos, me.pos);
          const dist = len2(to[0], to[1], to[2]);
          // FOV check vs current vel heading; if standing still, allow
          const forward: Vec3 =
            me.vel[0] === 0 && me.vel[2] === 0
              ? [0, 0, 1]
              : norm(me.vel[0], 0, me.vel[2]);
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
        desired = sub(target.pos, me.pos);
      } else {
        // wander to center while no target
        desired = mul(norm(-me.pos[0], 0, -me.pos[2]), 1);
      }

      // Separation
      let sepX = 0,
        sepZ = 0,
        neighbors = 0;
      for (const other of next) {
        if (other.id === me.id || !other.alive) continue;
        const dx = me.pos[0] - other.pos[0];
        const dz = me.pos[2] - other.pos[2];
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

      // Keep roughly inside arena bounds (soft)
      const radial = len2(me.pos[0], 0, me.pos[2]);
      if (radial > ARENA_RADIUS * 0.9) {
        const pull = mul(norm(-me.pos[0], 0, -me.pos[2]), 2.5);
        desired = add(desired, pull);
      }

      // Acceleration toward desired
      const desiredDir = norm(desired[0], 0, desired[2]);
      const targetVel = mul(desiredDir, BOT_SPEED);
      const dv = sub(targetVel, me.vel);
      const maxDv = BOT_ACCEL * dt;
      const dvLen = Math.hypot(dv[0], dv[2]);
      const apply = dvLen > maxDv ? mul(dv, maxDv / (dvLen || 1)) : dv;
      me.vel = add(me.vel, [apply[0], 0, apply[2]]);

      // Integrate & clamp to arena floor (y=0)
      me.pos = add(me.pos, mul(me.vel, dt));
      me.pos[1] = 0;

      // Attempt attack if in range & cooldown
      if (target) {
        const to = sub(target.pos, me.pos);
        const dist = len2(to[0], to[1], to[2]);
        if (dist <= ATTACK_RANGE && now - me.lastHitAt >= ATTACK_COOLDOWN) {
          target.hp -= ATTACK_DPS * ATTACK_COOLDOWN; // burst-y damage
          me.lastHitAt = now;
        }
      }

      // Death
      if (me.hp <= 0) {
        me.alive = false as unknown as boolean; // intentional TS trick to ensure boolean
        me.alive = false;
      }
    }

    // Write back
    setBots(next);
  });

  return { bots, zone, aliveCount, winner } as const;
}

// --- Renderables ------------------------------------------------------------
function Arena({ radius }: { radius: number }) {
  return (
    <group>
      {/* floor */}
      <mesh rotation-x={-Math.PI / 2} receiveShadow>
        <circleGeometry args={[radius, 64]} />
        <meshStandardMaterial roughness={1} metalness={0} color="#2a2a2a" />
      </mesh>

      {/* wall ring (visual) */}
      <mesh rotation-x={-Math.PI / 2} position={[0, 0.01, 0]}>
        <ringGeometry args={[radius * 0.98, radius, 64]} />
        <meshBasicMaterial color="#3a3a3a" />
      </mesh>
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

function Bot({ bot }: { bot: BotState }) {
  const ref = useRef<THREE.Mesh>(null!);
  useFrame(() => {
    if (!ref.current) return;
    ref.current.position.set(bot.pos[0], bot.pos[1] + 0.5, bot.pos[2]);
    ref.current.rotation.y = headingFromVel(bot.vel);
    const s = bot.alive ? 1 : 0.2;
    ref.current.scale.setScalar(s);
  });

  const color = bot.alive ? `hsl(${(bot.id * 137) % 360}deg 80% 60%)` : "#333";
  return (
    <group>
      <mesh ref={ref} castShadow>
        <capsuleGeometry args={[0.4, 0.9, 6, 12]} />
        <meshStandardMaterial color={color} />
      </mesh>
      {/* HP bar */}
      {bot.alive && (
        <Html
          position={[bot.pos[0], 1.6, bot.pos[2]]}
          center
          distanceFactor={12}
          style={{ pointerEvents: "none" }}
        >
          <div
            style={{
              width: 48,
              height: 6,
              background: "#222",
              borderRadius: 4,
            }}
          >
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
    </group>
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
  // const { bots, zone, aliveCount, winner } = useBattleRoyale();

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

        <TheWorld />

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
  const { bots, zone, aliveCount, winner } = useBattleRoyale();
  return (
    <>
    <group position={[0, 0, 0]}>
      <Arena radius={ARENA_RADIUS} />
      <Zone zoneRef={zone} />
      {bots.map((b) => (
        <Bot key={b.id} bot={b} />
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
