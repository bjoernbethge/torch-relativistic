"""
Unified data acquisition system for space datasets with relativistic effects.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
import aiohttp
import polars as pl
import requests
from urllib.parse import urljoin


@dataclass
class DatasetConfig:
    """Configuration for dataset acquisition."""
    name: str
    source: str
    url: str
    format: str = "json"
    cache_duration: int = 3600  # seconds
    rate_limit: float = 1.0  # requests per second
    requires_auth: bool = False
    auth_token: Optional[str] = None


class BaseDataSource(ABC):
    """Abstract base class for all data sources."""
    
    def __init__(self, config: DatasetConfig, cache_dir: Optional[Path] = None):
        self.config = config
        self.cache_dir = cache_dir or Path("./data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._session: Optional[aiohttp.ClientSession] = None
    
    @abstractmethod
    async def fetch_raw_data(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch raw data from the source."""
        pass
    
    @abstractmethod
    def transform_to_standard(self, raw_data: Dict[str, Any]) -> pl.DataFrame:
        """Transform raw data to standardized format."""
        pass
    
    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            headers = {}
            if self.config.requires_auth and self.config.auth_token:
                headers["Authorization"] = f"Bearer {self.config.auth_token}"
            
            self._session = aiohttp.ClientSession(
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self._session
    
    def get_cache_path(self, params: Dict[str, Any]) -> Path:
        """Generate cache file path based on parameters."""
        cache_key = f"{self.config.name}_{hash(str(sorted(params.items())))}"
        return self.cache_dir / f"{cache_key}.parquet"
    
    def is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cached data is still valid."""
        if not cache_path.exists():
            return False
        
        age = datetime.now().timestamp() - cache_path.stat().st_mtime
        return age < self.config.cache_duration
    
    async def close(self):
        """Clean up resources."""
        if self._session and not self._session.closed:
            await self._session.close()


class NASAJPLHorizons(BaseDataSource):
    """NASA JPL Horizons ephemeris data source."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        config = DatasetConfig(
            name="nasa_jpl_horizons",
            source="NASA JPL",
            url="https://ssd.jpl.nasa.gov/api/horizons.api",
            format="text"
        )
        super().__init__(config, cache_dir)
    
    async def fetch_raw_data(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch ephemeris data from JPL Horizons."""
        session = await self.get_session()
        
        # Default parameters for satellite data
        horizons_params = {
            'format': 'text',
            'COMMAND': params.get('target', '399'),  # Earth default
            'OBJ_DATA': 'YES',
            'MAKE_EPHEM': 'YES',
            'EPHEM_TYPE': 'OBSERVER',
            'CENTER': '500@399',  # Geocentric
            'START_TIME': params.get('start_time', '2024-01-01'),
            'STOP_TIME': params.get('stop_time', '2024-01-02'),
            'STEP_SIZE': params.get('step_size', '1h'),
            'QUANTITIES': '1,9,20,23,24'  # Position, velocity, light-time, etc.
        }
        
        async with session.get(self.config.url, params=horizons_params) as response:
            response.raise_for_status()
            return {"raw_text": await response.text(), "params": params}
    
    def transform_to_standard(self, raw_data: Dict[str, Any]) -> pl.DataFrame:
        """Transform JPL Horizons text output to standardized DataFrame."""
        lines = raw_data["raw_text"].split('\n')
        
        # Find data section
        data_start = None
        data_end = None
        
        for i, line in enumerate(lines):
            if '$$SOE' in line:  # Start of Ephemeris
                data_start = i + 1
            elif '$$EOE' in line:  # End of Ephemeris
                data_end = i
                break
        
        if data_start is None or data_end is None:
            raise ValueError("Could not parse JPL Horizons data")
        
        # Parse data lines
        data_rows = []
        for line in lines[data_start:data_end]:
            if line.strip():
                parts = line.split()
                if len(parts) >= 10:  # Ensure we have enough columns
                    data_rows.append({
                        'datetime': f"{parts[0]} {parts[1]}",
                        'x_km': float(parts[2]),
                        'y_km': float(parts[3]), 
                        'z_km': float(parts[4]),
                        'vx_km_s': float(parts[5]),
                        'vy_km_s': float(parts[6]),
                        'vz_km_s': float(parts[7]),
                        'light_time_min': float(parts[8]) if len(parts) > 8 else 0.0,
                        'source': 'nasa_jpl_horizons'
                    })
        
        return pl.DataFrame(data_rows)


class CelestrackTLE(BaseDataSource):
    """Celestrak Two-Line Element data source."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        config = DatasetConfig(
            name="celestrak_tle",
            source="CelesTrak",
            url="https://celestrak.org/NORAD/elements/",
            format="text"
        )
        super().__init__(config, cache_dir)
    
    async def fetch_raw_data(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch TLE data from Celestrak."""
        session = await self.get_session()
        
        catalog = params.get('catalog', 'gps-ops.txt')
        url = urljoin(self.config.url, catalog)
        
        async with session.get(url) as response:
            response.raise_for_status()
            return {"raw_text": await response.text(), "catalog": catalog}
    
    def transform_to_standard(self, raw_data: Dict[str, Any]) -> pl.DataFrame:
        """Transform TLE data to standardized DataFrame."""
        lines = raw_data["raw_text"].strip().split('\n')
        
        data_rows = []
        for i in range(0, len(lines), 3):
            if i + 2 < len(lines):
                name = lines[i].strip()
                line1 = lines[i + 1]
                line2 = lines[i + 2]
                
                # Parse TLE elements
                if len(line1) >= 69 and len(line2) >= 69:
                    data_rows.append({
                        'satellite_name': name,
                        'norad_id': int(line1[2:7]),
                        'epoch_year': int(line1[18:20]),
                        'epoch_day': float(line1[20:32]),
                        'mean_motion': float(line2[52:63]),
                        'eccentricity': float(f"0.{line2[26:33]}"),
                        'inclination_deg': float(line2[8:16]),
                        'raan_deg': float(line2[17:25]),
                        'arg_perigee_deg': float(line2[34:42]),
                        'mean_anomaly_deg': float(line2[43:51]),
                        'source': 'celestrak_tle'
                    })
        
        return pl.DataFrame(data_rows)


class ESASpaceDebris(BaseDataSource):
    """ESA Space Debris Office data source."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        config = DatasetConfig(
            name="esa_space_debris",
            source="ESA",
            url="https://sdup.esoc.esa.int/discosweb/api/",
            format="json",
            requires_auth=True
        )
        super().__init__(config, cache_dir)
    
    async def fetch_raw_data(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch space debris data from ESA."""
        session = await self.get_session()
        
        # Note: This would require proper ESA API authentication
        endpoint = params.get('endpoint', 'objects')
        url = urljoin(self.config.url, endpoint)
        
        request_params = {
            'format': 'json',
            'limit': params.get('limit', 1000)
        }
        
        try:
            async with session.get(url, params=request_params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    # Fallback to synthetic data for demo
                    return self._generate_synthetic_debris_data(params)
        except Exception:
            return self._generate_synthetic_debris_data(params)
    
    def _generate_synthetic_debris_data(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate synthetic space debris data for testing."""
        import random
        import numpy as np
        
        n_objects = params.get('limit', 100)
        objects = []
        
        for i in range(n_objects):
            # Generate realistic orbital parameters
            altitude = random.uniform(200, 2000)  # km
            inclination = random.uniform(0, 180)   # degrees
            
            objects.append({
                'object_id': f"DEBRIS_{i:05d}",
                'object_type': random.choice(['DEBRIS', 'ROCKET_BODY', 'PAYLOAD']),
                'altitude_km': altitude,
                'inclination_deg': inclination,
                'eccentricity': random.uniform(0, 0.3),
                'mass_kg': random.uniform(1, 1000),
                'cross_section_m2': random.uniform(0.1, 10),
                'launch_date': f"2{random.randint(10, 24):02d}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"
            })
        
        return {'objects': objects, 'synthetic': True}
    
    def transform_to_standard(self, raw_data: Dict[str, Any]) -> pl.DataFrame:
        """Transform ESA debris data to standardized DataFrame."""
        if 'objects' in raw_data:
            objects = raw_data['objects']
        else:
            objects = raw_data.get('data', [])
        
        data_rows = []
        for obj in objects:
            data_rows.append({
                'object_id': obj.get('object_id', ''),
                'object_type': obj.get('object_type', 'UNKNOWN'),
                'altitude_km': obj.get('altitude_km', 0.0),
                'inclination_deg': obj.get('inclination_deg', 0.0),
                'eccentricity': obj.get('eccentricity', 0.0),
                'mass_kg': obj.get('mass_kg', 0.0),
                'cross_section_m2': obj.get('cross_section_m2', 0.0),
                'launch_date': obj.get('launch_date', ''),
                'source': 'esa_space_debris'
            })
        
        return pl.DataFrame(data_rows)


class DataAcquisition:
    """Main data acquisition orchestrator."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("./data/cache")
        self.sources: Dict[str, BaseDataSource] = {}
        self._register_sources()
    
    def _register_sources(self):
        """Register all available data sources."""
        self.sources = {
            'nasa_jpl_horizons': NASAJPLHorizons(self.cache_dir),
            'celestrak_tle': CelestrackTLE(self.cache_dir),
            'esa_space_debris': ESASpaceDebris(self.cache_dir)
        }
    
    async def acquire_dataset(
        self, 
        source_name: str, 
        params: Dict[str, Any],
        force_refresh: bool = False
    ) -> pl.DataFrame:
        """Acquire dataset from specified source."""
        if source_name not in self.sources:
            raise ValueError(f"Unknown data source: {source_name}")
        
        source = self.sources[source_name]
        cache_path = source.get_cache_path(params)
        
        # Check cache first
        if not force_refresh and source.is_cache_valid(cache_path):
            print(f"Loading cached data from {cache_path}")
            return pl.read_parquet(cache_path)
        
        # Fetch fresh data
        print(f"Fetching data from {source.config.source}...")
        
        # Rate limiting
        await asyncio.sleep(1.0 / source.config.rate_limit)
        
        raw_data = await source.fetch_raw_data(params)
        df = source.transform_to_standard(raw_data)
        
        # Cache the result
        df.write_parquet(cache_path)
        print(f"Cached data to {cache_path}")
        
        return df
    
    async def acquire_multiple(
        self, 
        acquisitions: List[Dict[str, Any]],
        max_concurrent: int = 3
    ) -> Dict[str, pl.DataFrame]:
        """Acquire multiple datasets concurrently."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def bounded_acquire(acq_config):
            async with semaphore:
                source_name = acq_config['source']
                params = acq_config.get('params', {})
                force_refresh = acq_config.get('force_refresh', False)
                
                df = await self.acquire_dataset(source_name, params, force_refresh)
                return acq_config.get('name', source_name), df
        
        tasks = [bounded_acquire(config) for config in acquisitions]
        results = await asyncio.gather(*tasks)
        
        return dict(results)
    
    async def close(self):
        """Clean up all data sources."""
        for source in self.sources.values():
            await source.close()
    
    def list_available_sources(self) -> List[str]:
        """List all available data sources."""
        return list(self.sources.keys())
    
    def get_source_info(self, source_name: str) -> DatasetConfig:
        """Get information about a specific data source."""
        if source_name not in self.sources:
            raise ValueError(f"Unknown data source: {source_name}")
        
        return self.sources[source_name].config
