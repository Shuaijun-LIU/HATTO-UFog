"""CLI entrypoints for the ufog_network."""
from __future__ import annotations

import json
from pathlib import Path

import click

from ufog_network.config import load_config
from ufog_network.env.generators import generate_world
from ufog_network.io import export_world, import_world
from ufog_network.sim.simulator import Simulator
from ufog_network.baselines.td3 import train_td3
from ufog_network.control.ddqn import DDQNConfig, train_ddqn_gain_adjuster


@click.group()
def main() -> None:
    """UAV-assisted fog computing framework CLI."""
    pass


@main.command("run")
@click.option("--config", "config_path", default=None, help="Path to config yaml/json")
@click.option("--output", "output_root", default="runs", help="Output directory")
@click.option("--world", "world_path", default=None, help="Optional world.json to load")
@click.option("--no-export-world", is_flag=True, help="Skip exporting world.json")
def run_cmd(config_path: str | None, output_root: str, world_path: str | None, no_export_world: bool) -> None:
    cfg = load_config(config_path)
    world = import_world(world_path) if world_path else None
    sim = Simulator(cfg, world=world)
    result = sim.run(output_root=output_root, export_world_json=not no_export_world)
    click.echo(json.dumps(result.summary, indent=2))
    click.echo(f"Output: {result.output_dir}")


@main.command("generate-world")
@click.option("--config", "config_path", default=None, help="Path to config yaml/json")
@click.option("--output", "output_path", default="world.json", help="Output world.json")
def gen_world_cmd(config_path: str | None, output_path: str) -> None:
    cfg = load_config(config_path)
    world = generate_world(cfg.world)
    export_world(world, output_path)
    click.echo(f"World saved to {output_path}")


@main.command("train-td3")
@click.option("--config", "config_path", default=None, help="Path to config yaml/json")
@click.option("--output", "output_root", default="runs/td3_train", help="Output directory")
def train_td3_cmd(config_path: str | None, output_root: str) -> None:
    cfg = load_config(config_path)
    preset = (cfg.baseline.presets or {}).get("td3", {})
    params = {**preset, **(cfg.baseline.params or {})}
    summary = train_td3(cfg, output_root, params=params)
    click.echo(json.dumps(summary, indent=2))


@main.command("train-fear-pid")
@click.option("--config", "config_path", default=None, help="Path to config yaml/json")
@click.option("--output", "output_root", default="runs/fear_pid", help="Output directory")
def train_fear_pid_cmd(config_path: str | None, output_root: str) -> None:
    if config_path:
        cfg = load_config(config_path).fear_ddqn
    else:
        cfg = DDQNConfig()
    summary = train_ddqn_gain_adjuster(output_root, cfg)
    click.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
