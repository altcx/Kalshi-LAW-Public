#!/usr/bin/env python3
"""Service manager for running the daily scheduler as a background service."""

import sys
import os
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from loguru import logger
import psutil

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class ServiceManager:
    """Manages the daily scheduler as a background service."""
    
    def __init__(self):
        """Initialize service manager."""
        self.project_root = project_root
        self.pid_file = self.project_root / "logs" / "scheduler.pid"
        self.log_file = self.project_root / "logs" / "service_manager.log"
        self.scheduler_script = self.project_root / "src" / "automation" / "daily_scheduler.py"
        
        # Ensure directories exist
        self.pid_file.parent.mkdir(exist_ok=True)
        self.log_file.parent.mkdir(exist_ok=True)
        
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging for service manager."""
        logger.remove()
        
        # Console logging
        logger.add(
            sys.stdout,
            level="INFO",
            format="{time:HH:mm:ss} | {level} | {message}"
        )
        
        # File logging
        logger.add(
            self.log_file,
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            rotation="10 MB",
            retention="30 days"
        )
    
    def is_running(self) -> bool:
        """Check if the scheduler service is currently running."""
        if not self.pid_file.exists():
            return False
        
        try:
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())
            
            # Check if process with this PID exists and is our scheduler
            if psutil.pid_exists(pid):
                process = psutil.Process(pid)
                cmdline = ' '.join(process.cmdline())
                
                if 'daily_scheduler.py' in cmdline:
                    return True
                else:
                    # PID file exists but process is different, clean up
                    logger.warning(f"PID file exists but process {pid} is not our scheduler")
                    self.pid_file.unlink()
                    return False
            else:
                # PID file exists but process doesn't, clean up
                logger.warning(f"PID file exists but process {pid} not found")
                self.pid_file.unlink()
                return False
                
        except (ValueError, FileNotFoundError, psutil.NoSuchProcess) as e:
            logger.warning(f"Error checking if service is running: {e}")
            if self.pid_file.exists():
                self.pid_file.unlink()
            return False
    
    def start_service(self) -> bool:
        """Start the scheduler service in the background."""
        if self.is_running():
            logger.info("Scheduler service is already running")
            return True
        
        try:
            logger.info("Starting scheduler service...")
            
            # Start the scheduler as a background process
            process = subprocess.Popen(
                [sys.executable, str(self.scheduler_script)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.project_root),
                start_new_session=True  # Detach from parent process
            )
            
            # Give it a moment to start
            time.sleep(2)
            
            # Check if it's still running
            if process.poll() is None:
                # Save PID
                with open(self.pid_file, 'w') as f:
                    f.write(str(process.pid))
                
                logger.info(f"Scheduler service started with PID {process.pid}")
                return True
            else:
                # Process exited immediately
                stdout, stderr = process.communicate()
                logger.error(f"Scheduler failed to start:")
                logger.error(f"STDOUT: {stdout.decode()}")
                logger.error(f"STDERR: {stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"Error starting scheduler service: {e}")
            return False
    
    def stop_service(self) -> bool:
        """Stop the scheduler service."""
        if not self.is_running():
            logger.info("Scheduler service is not running")
            return True
        
        try:
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())
            
            logger.info(f"Stopping scheduler service (PID {pid})...")
            
            # Try graceful shutdown first
            process = psutil.Process(pid)
            process.terminate()
            
            # Wait up to 10 seconds for graceful shutdown
            try:
                process.wait(timeout=10)
                logger.info("Scheduler service stopped gracefully")
            except psutil.TimeoutExpired:
                # Force kill if graceful shutdown failed
                logger.warning("Graceful shutdown timed out, force killing...")
                process.kill()
                process.wait()
                logger.info("Scheduler service force killed")
            
            # Clean up PID file
            if self.pid_file.exists():
                self.pid_file.unlink()
            
            return True
            
        except (FileNotFoundError, ValueError, psutil.NoSuchProcess) as e:
            logger.warning(f"Error stopping service: {e}")
            # Clean up PID file anyway
            if self.pid_file.exists():
                self.pid_file.unlink()
            return True
        except Exception as e:
            logger.error(f"Error stopping scheduler service: {e}")
            return False
    
    def restart_service(self) -> bool:
        """Restart the scheduler service."""
        logger.info("Restarting scheduler service...")
        
        if not self.stop_service():
            logger.error("Failed to stop service for restart")
            return False
        
        # Wait a moment between stop and start
        time.sleep(2)
        
        return self.start_service()
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get detailed status of the scheduler service."""
        status = {
            'running': False,
            'pid': None,
            'uptime': None,
            'memory_usage': None,
            'cpu_percent': None,
            'last_log_entry': None
        }
        
        if self.is_running():
            try:
                with open(self.pid_file, 'r') as f:
                    pid = int(f.read().strip())
                
                process = psutil.Process(pid)
                status.update({
                    'running': True,
                    'pid': pid,
                    'uptime': datetime.now() - datetime.fromtimestamp(process.create_time()),
                    'memory_usage': process.memory_info().rss / 1024 / 1024,  # MB
                    'cpu_percent': process.cpu_percent()
                })
                
            except Exception as e:
                logger.warning(f"Error getting process details: {e}")
                status['running'] = False
        
        # Get last log entry
        try:
            scheduler_log = self.project_root / "logs" / "daily_scheduler.log"
            if scheduler_log.exists():
                with open(scheduler_log, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        status['last_log_entry'] = lines[-1].strip()
        except Exception as e:
            logger.warning(f"Error reading scheduler log: {e}")
        
        return status
    
    def run_task_once(self, task: str) -> bool:
        """Run a specific scheduler task once."""
        try:
            logger.info(f"Running task once: {task}")
            
            result = subprocess.run(
                [sys.executable, str(self.scheduler_script), "--run-once", task],
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                logger.info(f"Task '{task}' completed successfully")
                if result.stdout:
                    logger.info(f"Output: {result.stdout}")
                return True
            else:
                logger.error(f"Task '{task}' failed with return code {result.returncode}")
                if result.stderr:
                    logger.error(f"Error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"Task '{task}' timed out after 5 minutes")
            return False
        except Exception as e:
            logger.error(f"Error running task '{task}': {e}")
            return False
    
    def monitor_service(self, check_interval: int = 300):
        """Monitor the service and restart if it crashes."""
        logger.info(f"Starting service monitor (check interval: {check_interval}s)")
        
        try:
            while True:
                if not self.is_running():
                    logger.warning("Scheduler service is not running, attempting to restart...")
                    if self.start_service():
                        logger.info("Service restarted successfully")
                    else:
                        logger.error("Failed to restart service")
                
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            logger.info("Service monitor stopped by user")
        except Exception as e:
            logger.error(f"Error in service monitor: {e}")


def main():
    """Main entry point for service manager."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Service manager for daily scheduler")
    parser.add_argument("command", choices=["start", "stop", "restart", "status", "monitor"],
                       help="Service command to execute")
    parser.add_argument("--task", choices=["morning", "evening", "prediction", "alerts", "health", "retry"],
                       help="Run a specific task once (use with 'start' command)")
    parser.add_argument("--monitor-interval", type=int, default=300,
                       help="Monitor check interval in seconds (default: 300)")
    
    args = parser.parse_args()
    
    manager = ServiceManager()
    
    if args.command == "start":
        if args.task:
            # Run specific task once
            success = manager.run_task_once(args.task)
            return 0 if success else 1
        else:
            # Start service
            success = manager.start_service()
            return 0 if success else 1
    
    elif args.command == "stop":
        success = manager.stop_service()
        return 0 if success else 1
    
    elif args.command == "restart":
        success = manager.restart_service()
        return 0 if success else 1
    
    elif args.command == "status":
        status = manager.get_service_status()
        
        print(f"Scheduler Service Status:")
        print(f"  Running: {'Yes' if status['running'] else 'No'}")
        
        if status['running']:
            print(f"  PID: {status['pid']}")
            print(f"  Uptime: {status['uptime']}")
            print(f"  Memory Usage: {status['memory_usage']:.1f} MB")
            print(f"  CPU Usage: {status['cpu_percent']:.1f}%")
        
        if status['last_log_entry']:
            print(f"  Last Log: {status['last_log_entry']}")
        
        return 0
    
    elif args.command == "monitor":
        try:
            manager.monitor_service(args.monitor_interval)
            return 0
        except KeyboardInterrupt:
            return 0
        except Exception as e:
            logger.error(f"Monitor failed: {e}")
            return 1
    
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())