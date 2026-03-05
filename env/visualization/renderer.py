"""
Pygame 2D renderer for the Ocean IoT SAGIN simulation.

Features:
  - Top-down ocean map with coastline
  - Colour-coded node icons with labels
  - Communication link lines
  - Sidebar with live statistics
  - Headless mode for cloud servers (no display)
"""

from __future__ import annotations
import os
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..ocean_env import OceanEnv

from ..config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, MAP_DISPLAY_WIDTH, MAP_DISPLAY_HEIGHT,
    SIDEBAR_WIDTH, MAP_WIDTH, MAP_HEIGHT, COASTLINE_X, FPS,
    COLOR_OCEAN, COLOR_COASTLINE, COLOR_LINK,
    COLOR_SIDEBAR_BG, COLOR_TEXT, NODE_COLORS,
)

_HEADLESS = os.environ.get("SDL_VIDEODRIVER", "") == "dummy"


def _world_to_screen(pos, map_w, map_h, disp_w, disp_h):
    sx = pos[0] / map_w * disp_w
    sy = (1.0 - pos[1] / map_h) * disp_h
    return int(sx), int(sy)


class PygameRenderer:

    def __init__(self, env: "OceanEnv"):
        import pygame
        self.pg = pygame
        if _HEADLESS:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("SAGIN Ocean IoT MEC Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 14)
        self.font_title = pygame.font.SysFont("consolas", 18, bold=True)
        self.env = env

    # ------------------------------------------------------------------

    def draw(self):
        pg = self.pg
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.close()
                return

        self.screen.fill((0, 0, 0))
        self._draw_map()
        self._draw_links()
        self._draw_nodes()
        self._draw_sidebar()
        pg.display.flip()
        self.clock.tick(FPS)

    # ------------------------------------------------------------------

    def _draw_map(self):
        pg = self.pg
        # Ocean
        pg.draw.rect(self.screen, COLOR_OCEAN,
                      (0, 0, MAP_DISPLAY_WIDTH, MAP_DISPLAY_HEIGHT))
        # Coastline strip
        cx = int(COASTLINE_X / MAP_WIDTH * MAP_DISPLAY_WIDTH)
        pg.draw.rect(self.screen, COLOR_COASTLINE,
                      (0, 0, cx, MAP_DISPLAY_HEIGHT))
        # Grid lines (every 50 km)
        for km in range(0, 250, 50):
            gx = int(km * 1000 / MAP_WIDTH * MAP_DISPLAY_WIDTH)
            gy = int((1.0 - km * 1000 / MAP_HEIGHT) * MAP_DISPLAY_HEIGHT)
            pg.draw.line(self.screen, (30, 60, 110),
                         (gx, 0), (gx, MAP_DISPLAY_HEIGHT), 1)
            if 0 <= gy <= MAP_DISPLAY_HEIGHT:
                pg.draw.line(self.screen, (30, 60, 110),
                             (0, gy), (MAP_DISPLAY_WIDTH, gy), 1)

    def _draw_links(self):
        pg = self.pg
        edges = self.env.topology_mgr.get_edges()
        id_map = {n.id: n for n in self.env.nodes}
        for a_id, b_id, _ in edges:
            a = id_map.get(a_id)
            b = id_map.get(b_id)
            if a is None or b is None:
                continue
            sa = _world_to_screen(a.position, MAP_WIDTH, MAP_HEIGHT,
                                  MAP_DISPLAY_WIDTH, MAP_DISPLAY_HEIGHT)
            sb = _world_to_screen(b.position, MAP_WIDTH, MAP_HEIGHT,
                                  MAP_DISPLAY_WIDTH, MAP_DISPLAY_HEIGHT)
            pg.draw.line(self.screen, COLOR_LINK, sa, sb, 1)

    def _draw_nodes(self):
        pg = self.pg
        for node in self.env.nodes:
            sx, sy = _world_to_screen(node.position, MAP_WIDTH, MAP_HEIGHT,
                                      MAP_DISPLAY_WIDTH, MAP_DISPLAY_HEIGHT)
            color = NODE_COLORS.get(node.node_type, (255, 255, 255))
            ntype = node.node_type

            if ntype == "satellite":
                # Triangle
                pts = [(sx, sy - 8), (sx - 6, sy + 5), (sx + 6, sy + 5)]
                pg.draw.polygon(self.screen, color, pts)
            elif ntype == "uav":
                # Diamond
                pts = [(sx, sy - 7), (sx + 7, sy), (sx, sy + 7), (sx - 7, sy)]
                pg.draw.polygon(self.screen, color, pts)
            elif ntype == "ship":
                pg.draw.rect(self.screen, color, (sx - 5, sy - 5, 10, 10))
            elif ntype == "buoy":
                pg.draw.circle(self.screen, color, (sx, sy), 4)
            else:
                # base_station – hexagon
                pts = []
                for k in range(6):
                    angle = math.pi / 6 + k * math.pi / 3
                    pts.append((sx + int(8 * math.cos(angle)),
                                sy + int(8 * math.sin(angle))))
                pg.draw.polygon(self.screen, color, pts)

            label = self.font.render(f"{node.id}", True, color)
            self.screen.blit(label, (sx + 8, sy - 6))

    def _draw_sidebar(self):
        pg = self.pg
        x0 = MAP_DISPLAY_WIDTH
        pg.draw.rect(self.screen, COLOR_SIDEBAR_BG,
                      (x0, 0, SIDEBAR_WIDTH, SCREEN_HEIGHT))

        lines = [
            "=== SAGIN Simulation ===",
            "",
            f"Step: {self.env.step_count} / {self.env.episode_length}",
            f"Noise factor: {self.env.noise_factor:.2f}",
            "",
        ]

        # Node counts by type
        from collections import Counter
        ctr = Counter(n.node_type for n in self.env.nodes)
        lines.append("-- Nodes --")
        for ntype in ["satellite", "uav", "ship", "buoy", "base_station"]:
            lines.append(f"  {ntype}: {ctr.get(ntype, 0)}")
        lines.append(f"  TOTAL: {len(self.env.nodes)}")
        lines.append("")

        edges = self.env.topology_mgr.get_edges()
        lines.append("-- Topology --")
        lines.append(f"  Links: {len(edges)}")
        mec_n = sum(1 for n in self.env.nodes if n.is_mec)
        lines.append(f"  MEC nodes: {mec_n}")
        lines.append(f"  Pending tasks: {len(self.env.mec_mgr.pending_tasks)}")
        lines.append("")

        # Avg degree
        degrees = [self.env.topology_mgr.get_node_degree(n.id)
                    for n in self.env.nodes]
        avg_deg = sum(degrees) / max(len(degrees), 1)
        lines.append(f"  Avg degree: {avg_deg:.1f}")

        # Legend
        lines += ["", "-- Legend --"]
        for ntype, color in NODE_COLORS.items():
            lines.append(f"  {ntype}")

        y = 15
        for i, line in enumerate(lines):
            if line.startswith("==="):
                surf = self.font_title.render(line, True, COLOR_TEXT)
            else:
                surf = self.font.render(line, True, COLOR_TEXT)
            self.screen.blit(surf, (x0 + 10, y))
            y += 20

    # ------------------------------------------------------------------

    def close(self):
        self.pg.quit()
