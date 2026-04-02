#!/usr/bin/env python3
"""
Automated video recording script for RL agent demonstration
Uses FFmpeg to record the screen while running the game demo
"""

import subprocess
import time
import os
import argparse
import signal
import sys
from pathlib import Path


def get_display():
    """Get the X11 display"""
    display = os.environ.get('DISPLAY', ':0.0')
    return display


def get_resolution():
    """Detect screen resolution"""
    try:
        result = subprocess.run(['xrandr', '--current'], capture_output=True, text=True)
        # Look for the primary display resolution
        for line in result.stdout.split('\n'):
            if 'connected primary' in line or 'connected' in line:
                parts = line.split()
                for part in parts:
                    if 'x' in part and '+' in part:
                        return part.split('+')[0]
    except:
        pass
    return "1920x1080"  # Fallback


def record_demo(algorithm='ppo', episodes=3, output_file='videos/demo.mp4', fps=30, quality='medium'):
    """
    Record the RL agent demo to a video file
    
    Args:
        algorithm: 'ppo' or 'dqn'
        episodes: number of episodes to record
        output_file: output video file path
        fps: frames per second
        quality: 'high', 'medium', 'low'
    """
    
    # Create output directory
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Quality settings
    quality_settings = {
        'high': {'crf': 18, 'preset': 'slow'},
        'medium': {'crf': 23, 'preset': 'medium'},
        'low': {'crf': 28, 'preset': 'veryfast'},
    }
    
    settings = quality_settings.get(quality, quality_settings['medium'])
    crf = settings['crf']
    preset = settings['preset']
    
    # Get display and resolution
    display = get_display()
    resolution = get_resolution()
    
    print("=" * 80)
    print("🎬 AUTOMATED RL AGENT VIDEO RECORDING")
    print("=" * 80)
    print(f"📺 Display:       {display}")
    print(f"📏 Resolution:    {resolution}")
    print(f"🎬 Output:        {output_file}")
    print(f"🎞️  FPS:           {fps}")
    print(f"⚙️  Quality:       {quality} (CRF: {crf}, Preset: {preset})")
    print(f"🤖 Algorithm:     {algorithm.upper()}")
    print(f"📊 Episodes:      {episodes}")
    print("=" * 80)
    print("\n⏳ Starting recording in 3 seconds...\n")
    
    time.sleep(3)
    
    # Start FFmpeg recording
    ffmpeg_cmd = [
        'ffmpeg',
        '-video_size', resolution,
        '-framerate', str(fps),
        '-f', 'x11grab',
        '-i', display,
        '-c:v', 'libx264',
        '-preset', preset,
        '-crf', str(crf),
        '-pix_fmt', 'yuv420p',
        str(output_file)
    ]
    
    print(f"🎥 Recording FFmpeg process started...")
    print(f"   Command: {' '.join(ffmpeg_cmd)}\n")
    
    try:
        # Start FFmpeg in background
        ffmpeg_process = subprocess.Popen(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE
        )
        
        # Give FFmpeg time to initialize
        time.sleep(2)
        
        # Run the game demo
        print(f"\n🎮 Starting game demo ({algorithm.upper()}, {episodes} episodes)...\n")
        
        demo_cmd = [
            'python3',
            'game_demo.py',
            '--algorithm', algorithm,
            '--episodes', str(episodes)
        ]
        
        demo_process = subprocess.Popen(demo_cmd)
        demo_process.wait()
        
        # Give video a moment to finalize
        print("\n⏳ Finalizing video recording...")
        time.sleep(2)
        
        # Stop FFmpeg gracefully
        if ffmpeg_process.stdin:
            ffmpeg_process.stdin.write(b'q')
            ffmpeg_process.stdin.flush()
        ffmpeg_process.wait(timeout=10)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Recording interrupted by user")
        ffmpeg_process.terminate()
        demo_process.terminate()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during recording: {e}")
        try:
            ffmpeg_process.terminate()
            demo_process.terminate()
        except:
            pass
        sys.exit(1)
    
    # Verify output
    if Path(output_file).exists():
        file_size = Path(output_file).stat().st_size / (1024*1024)  # MB
        print("\n" + "=" * 80)
        print(f"✅ VIDEO RECORDING COMPLETE!")
        print("=" * 80)
        print(f"📁 Output File:   {output_file}")
        print(f"📊 File Size:     {file_size:.2f} MB")
        print(f"🎥 Duration:      ~{episodes * 15} seconds (approx)")
        print("=" * 80)
        print("\n🎬 Next steps:")
        print("  1. Play video: ffplay " + output_file)
        print("  2. Or: vlc " + output_file)
        print("  3. Include in your submission")
        print("\n")
    else:
        print(f"\n❌ Error: Output file {output_file} was not created")
        sys.exit(1)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Automated RL Agent Video Recording",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Record PPO demo with 3 episodes at medium quality
  python3 record_video.py --algorithm ppo --episodes 3
  
  # Record DQN demo at high quality
  python3 record_video.py --algorithm dqn --episodes 5 --quality high
  
  # Custom output location
  python3 record_video.py --algorithm ppo --output my_demo.mp4
        """
    )
    
    parser.add_argument('--algorithm', type=str, default='ppo',
                       choices=['ppo', 'dqn'],
                       help='Algorithm to demonstrate (default: ppo)')
    parser.add_argument('--episodes', type=int, default=3,
                       help='Number of episodes (default: 3)')
    parser.add_argument('--output', type=str, default='videos/demo_ppo.mp4',
                       help='Output video file (default: videos/demo_ppo.mp4)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second (default: 30)')
    parser.add_argument('--quality', type=str, default='medium',
                       choices=['high', 'medium', 'low'],
                       help='Video quality (default: medium)')
    
    args = parser.parse_args()
    
    record_demo(
        algorithm=args.algorithm,
        episodes=args.episodes,
        output_file=args.output,
        fps=args.fps,
        quality=args.quality
    )


if __name__ == "__main__":
    main()
