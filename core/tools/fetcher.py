try:
    import requests
except Exception:  # pragma: no cover - optional dependency
    requests = None
import json
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import re
import platform
import logging
import shutil
try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

logger = logging.getLogger(__name__)

class DataFetcher:
    def __init__(self, cache_dir: str = "output/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def fetch_data(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Dynamically fetch data based on query analysis"""
        # Check cache first
        cache_key = self._get_cache_key(query, context["type"])
        cached = self._get_cached(cache_key, context["type"])
        if cached:
            return cached
            
        # Generate fetch plan based on query and context
        fetch_plan = self._generate_fetch_plan(query, context)
        
        try:
            # Execute fetch plan
            result = self._execute_fetch_plan(fetch_plan)
            
            # Cache successful result
            if "error" not in result:
                self._cache_result(cache_key, result)
            
            return result
        except Exception as e:
            return {
                "error": str(e),
                "status": "failed",
                "query": query,
                "type": context["type"]
            }
    
    def _get_cache_key(self, query: str, data_type: str) -> str:
        """Generate cache key from query and type"""
        # Extract key terms and normalize
        key_terms = [w.lower() for w in query.split() if len(w) > 3]
        return f"{data_type}_{'.'.join(key_terms)}"
        
    def _get_cached(self, key: str, data_type: str) -> Optional[Dict]:
        """Get cached result if fresh"""
        cache_file = self.cache_dir / f"{key}.json"
        if not cache_file.exists():
            return None
            
        data = json.loads(cache_file.read_text())
        cached_time = datetime.fromisoformat(data["timestamp"])
        
        # Check if cache is still fresh based on data type
        ttl = self._get_cache_ttl(data_type)
        if datetime.now() - cached_time > ttl:
            return None
            
        return data
        
    def _get_cache_ttl(self, data_type: str) -> timedelta:
        """Get TTL based on data type"""
        ttls = {
            "price": timedelta(minutes=5),
            "crypto": timedelta(minutes=5),
            "weather": timedelta(hours=1),
            "news": timedelta(hours=4),
            "network": timedelta(hours=12),
            "status": timedelta(minutes=5),
            "market": timedelta(minutes=15),
            "time": timedelta(minutes=1)
        }
        return ttls.get(data_type, timedelta(hours=1))
        
    def _cache_result(self, key: str, result: Dict):
        """Cache fetch result with timestamp"""
        cache_file = self.cache_dir / f"{key}.json"
        result["timestamp"] = datetime.now().isoformat()
        cache_file.write_text(json.dumps(result))
        
    def _generate_fetch_plan(self, query: str, context: Dict) -> Dict:
        """Generate fetch plan based on query analysis"""
        data_type = context["type"]
        
        if data_type == "crypto":
            return {
                "method": "api",
                "url": "https://api.coingecko.com/api/v3/simple/price",
                "params": {"ids": "bitcoin", "vs_currencies": "usd"},
                "type": "price"
            }
            
        elif data_type == "weather":
            # Extract location from query or use default
            location_match = re.search(r"(?:in|at|for)\s+([a-zA-Z\s,]+)", query)
            location = location_match.group(1) if location_match else "London"
            
            return {
                "method": "api",
                "url": "https://api.openweathermap.org/data/2.5/weather",
                "params": {
                    "q": location,
                    "appid": "YOUR_API_KEY",  # Would be fetched from config
                    "units": "metric"
                },
                "type": "weather"
            }
            
        elif data_type == "network":
            return {
                "method": "api",
                "url": "https://api.ipify.org",
                "params": {"format": "json"},
                "type": "network"
            }
            
        # Add more fetch plans for other data types...
        
        return {
            "method": "error",
            "error": f"Unsupported data type: {data_type}"
        }
        
    def _execute_fetch_plan(self, plan: Dict) -> Dict:
        """Execute a fetch plan with response validation"""
        method = plan.get("method")
        
        if method == "api":
            if requests is None:
                return {"type": "error", "error": "requests unavailable", "source": plan.get("url")}
            try:
                response = requests.get(plan["url"], params=plan.get("params"))
                response.raise_for_status()  # Raise for 4XX/5XX errors
                
                # Parse JSON response
                data = response.json()
                
                # Validate crypto price response
                if plan["type"] == "price" and "bitcoin" in str(plan.get("params", {})):
                    if not isinstance(data, dict):
                        raise ValueError("Invalid price data format")
                    if "bitcoin" not in data or "usd" not in data["bitcoin"]:
                        raise ValueError("Missing price data fields")
                        
                return {
                    "type": plan["type"],
                    "data": data,  # Already JSON parsed
                    "source": plan["url"]
                }
                
            except requests.RequestException as e:
                return {
                    "type": "error",
                    "error": f"API request failed: {str(e)}",
                    "source": plan["url"]
                }
            except (json.JSONDecodeError, ValueError) as e:
                return {
                    "type": "error", 
                    "error": f"Invalid response data: {str(e)}",
                    "source": plan["url"]
                }
        else:
            return {
                "error": f"Unsupported method: {method}",
                "status": "failed",
                "query": plan["query"],
                "type": plan["type"]
            } 

    def web_search(self, query: str) -> dict:
        """Perform a web search query and return results"""
        try:
            if requests is None:
                return {"status": "error", "output": "requests library not available"}

            # DuckDuckGo lite HTML endpoint (no JS required)
            url = "https://duckduckgo.com/html/"
            resp = requests.post(url, data={"q": query}, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()

            # VERY lightweight parsing – extract result titles & URLs via regex
            import re, html
            results = []
            for m in re.finditer(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', resp.text):
                href = html.unescape(m.group(1))
                title = re.sub(r"<[^>]+>", "", m.group(2))
                results.append(f"{title} – {href}")
                if len(results) >= 5:
                    break
            if not results:
                results.append("No results found or parsing failed")

            return {
                "status": "success",
                "output": "\n".join(results)
            }
        except Exception as e:
            return {
                "status": "error",
                "output": str(e)
            }

class SystemFetcher:
    def __init__(self):
        self.is_windows = platform.system().lower() == "windows"
        self.is_linux = platform.system().lower() == "linux"
        self.sudo_path = shutil.which("sudo")
        self.last_check = {}  # Cache check results
        
    def run_elevated(self, cmd: str, timeout: int = 30) -> Dict[str, Any]:
        """Run command with elevated privileges safely
        
        Args:
            cmd: Command to execute
            timeout: Maximum execution time in seconds
            
        Returns:
            Dict containing status, output, and error info
        """
        try:
            if self.is_windows:
                # Windows elevation via PowerShell
                ps_cmd = f'Start-Process -FilePath "cmd.exe" -ArgumentList "/c {cmd}" -Verb RunAs -Wait'
                result = subprocess.run(
                    ["powershell.exe", "-Command", ps_cmd],
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
            else:
                # Unix-like systems using sudo if available
                if self.sudo_path:
                    result = subprocess.run(
                        f"{self.sudo_path} {cmd}",
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=timeout
                    )
                else:
                    return {
                        "status": "failed",
                        "error": "Elevated privileges required but sudo not available"
                    }
                    
            return {
                "status": "success" if result.returncode == 0 else "failed",
                "output": result.stdout.strip(),
                "error": result.stderr.strip(),
                "code": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                "status": "failed",
                "error": f"Command timed out after {timeout} seconds"
            }
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }

    def get_kernel_info(self) -> Dict[str, Any]:
        """Get detailed kernel and system information"""
        try:
            if self.is_windows:
                cmd = "Get-WmiObject -Class Win32_OperatingSystem | ConvertTo-Json"
                result = subprocess.run(
                    ["powershell.exe", "-Command", cmd],
                    capture_output=True,
                    text=True
                )
                return json.loads(result.stdout)
            else:
                # Linux kernel info
                info = {}
                
                # Get kernel version
                with open("/proc/version", "r") as f:
                    info["kernel_version"] = f.read().strip()
                    
                # Get memory info
                with open("/proc/meminfo", "r") as f:
                    mem_info = {}
                    for line in f:
                        key, value = line.split(":", 1)
                        mem_info[key.strip()] = value.strip()
                    info["memory"] = mem_info
                    
                # Get CPU info
                with open("/proc/cpuinfo", "r") as f:
                    info["cpu"] = f.read().strip()
                    
                return info
                
        except Exception as e:
            logger.error(f"Failed to get kernel info: {e}")
            return {"error": str(e)}

    def get_system_telemetry(self) -> Dict[str, Any]:
        """Get real-time system performance metrics"""
        try:
            metrics = {}
            
            if self.is_windows:
                # Windows performance counters
                ps_cmd = r"""
                Get-Counter -Counter @(
                    '\Processor(_Total)\% Processor Time',
                    '\Memory\Available MBytes',
                    '\PhysicalDisk(_Total)\Disk Reads/sec',
                    '\PhysicalDisk(_Total)\Disk Writes/sec'
                ) | ConvertTo-Json
                """
                result = subprocess.run(
                    ["powershell.exe", "-Command", ps_cmd],
                    capture_output=True,
                    text=True
                )
                metrics = json.loads(result.stdout)
            else:
                # Linux system metrics
                metrics["cpu"] = {}
                metrics["memory"] = {}
                metrics["disk"] = {}
                
                # CPU usage
                result = subprocess.run(
                    ["top", "-bn1"], 
                    capture_output=True, 
                    text=True
                )
                metrics["cpu"]["usage"] = result.stdout
                
                # Memory usage
                result = subprocess.run(
                    ["free", "-m"],
                    capture_output=True,
                    text=True
                )
                metrics["memory"]["usage"] = result.stdout
                
                # Disk I/O
                result = subprocess.run(
                    ["iostat", "-x", "1", "1"],
                    capture_output=True,
                    text=True
                )
                metrics["disk"]["io"] = result.stdout
                
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get system telemetry: {e}")
            return {"error": str(e)}

    def get_network_info(self) -> Dict[str, Any]:
        """Get network interface and connection information"""
        try:
            if self.is_windows:
                cmd = "Get-NetAdapter | ConvertTo-Json"
                result = subprocess.run(
                    ["powershell.exe", "-Command", cmd],
                    capture_output=True,
                    text=True
                )
                return json.loads(result.stdout)
            else:
                result = subprocess.run(
                    ["ip", "addr"],
                    capture_output=True,
                    text=True
                )
                return {"interfaces": result.stdout}
                
        except Exception as e:
            logger.error(f"Failed to get network info: {e}")
            return {"error": str(e)}

    def get_full_host_status(self) -> Dict[str, Any]:
        """Get comprehensive host system status including hardware and drivers
        
        Returns:
            Dict containing parsed information from various WMI classes,
            organized by component type
        """
        status = {}
        wmi_queries = {
            "os": "Win32_OperatingSystem",
            "cpu": "Win32_Processor", 
            "ram": "Win32_PhysicalMemory",
            "disk": "Win32_LogicalDisk",
            "drivers": "Win32_SystemDriver"
        }
        
        for component, wmi_class in wmi_queries.items():
            try:
                cmd = f"Get-WmiObject {wmi_class} | ConvertTo-Json"
                result = self.run_elevated(cmd)
                
                if result["status"] == "success":
                    try:
                        # Handle both single object and array responses
                        parsed = json.loads(result["output"])
                        if isinstance(parsed, list):
                            # Multiple items (e.g. multiple CPUs or disks)
                            status[component] = parsed
                        else:
                            # Single item response
                            status[component] = [parsed]
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse {component} WMI data: {e}")
                        status[component] = {
                            "error": "JSON parse failed",
                            "raw_output": result["output"]
                        }
                else:
                    logger.warning(f"WMI query failed for {component}: {result.get('error')}")
                    status[component] = {
                        "error": f"WMI query failed: {result.get('error')}",
                        "raw_stderr": result.get("stderr", "")
                    }
                    
            except Exception as e:
                logger.error(f"Failed to get {component} info: {e}")
                status[component] = {
                    "error": f"Query failed: {str(e)}"
                }
        
        # Add timestamp
        status["timestamp"] = datetime.now().isoformat()
        
        # Add summary metrics
        try:
            summary = {
                "total_ram": sum(item.get("Capacity", 0) for item in status.get("ram", [])),
                "free_disk_space": sum(
                    disk.get("FreeSpace", 0) 
                    for disk in status.get("disk", [])
                ),
                "cpu_count": len(status.get("cpu", [])),
                "driver_count": len(status.get("drivers", [])),
            }
            status["summary"] = summary
        except Exception as e:
            logger.warning(f"Failed to generate summary metrics: {e}")
        
        return status 

    def get_driver_status(self) -> List[Dict[str, Any]]:
        """Get status of system services/drivers"""
        try:
            if platform.system() == "Windows":
                # Get service details with dependencies and description
                cmd = r"""
                Get-Service | Select-Object @{
                    Name='Name';Expression={$_.Name}
                }, @{
                    Name='DisplayName';Expression={$_.DisplayName}
                }, @{
                    Name='Status';Expression={$_.Status}
                }, @{
                    Name='StartType';Expression={$_.StartType}
                }, @{
                    Name='Dependencies';Expression={$_.DependentServices.Name -join ','}
                } | ConvertTo-Json
                """
                result = subprocess.run(
                    ["powershell.exe", "-Command", cmd],
                    capture_output=True,
                    text=True,
                    timeout=30  # Prevent hanging
                )
                
                if result.returncode == 0:
                    services = json.loads(result.stdout)
                    # Add severity assessment
                    for svc in services:
                        svc["Severity"] = (
                            "critical" if svc.get("StartType") == "Automatic" 
                            and svc.get("Status") != "Running"
                            else "warning" if svc.get("Status") != "Running"
                            else "info"
                        )
                    return services
                    
            elif platform.system() == "Linux":
                # Get systemd service status
                cmd = "systemctl list-units --type=service --all --plain --no-legend"
                result = subprocess.run(
                    cmd.split(),
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    services = []
                    for line in result.stdout.splitlines():
                        parts = line.split(None, 4)
                        if len(parts) >= 4:
                            name, load, active, sub, *_ = parts
                            services.append({
                                "Name": name,
                                "Status": active,
                                "SubState": sub,
                                "StartType": load,
                                "Severity": (
                                    "critical" if load == "loaded" 
                                    and active != "active"
                                    else "warning" if active != "active"
                                    else "info"
                                )
                            })
                    return services
                    
        except Exception as e:
            logger.error(f"Service status check failed: {e}")
            
        return []

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            "timestamp": datetime.now().isoformat(),
            "summary": {},
            "details": {}
        }

        if psutil is None:
            return {"error": "psutil not installed. Please install via pip"}

        try:
            # Get disk space
            disk_info = psutil.disk_usage('/')
            status["summary"]["free_disk_space"] = disk_info.free
            status["summary"]["total_disk_space"] = disk_info.total
            
            # Get memory info
            mem = psutil.virtual_memory()
            status["summary"]["free_ram"] = mem.available
            status["summary"]["total_ram"] = mem.total
            
            # Get CPU load
            status["summary"]["cpu_percent"] = psutil.cpu_percent(interval=1)
            
            # Get service status
            status["drivers"] = self.get_driver_status()
            
            # Add detailed metrics
            status["details"]["disk"] = {
                "used_percent": disk_info.percent,
                "mountpoint": "/"
            }
            
            status["details"]["memory"] = {
                "used_percent": mem.percent,
                "swap_used": psutil.swap_memory().used
            }
            
            return status
            
        except Exception as e:
            logger.error(f"System status check failed: {e}")
            return {"error": str(e)} 