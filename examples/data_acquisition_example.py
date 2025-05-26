"""
Example of using the torch_relativistic data acquisition system.
"""

import asyncio
from pathlib import Path
from torch_relativistic.data import DataAcquisition, SpaceDataLoader, RelativisticDataProcessor, SpaceDataVisualizer

async def main():
    """Main example function."""
    
    # Initialize data acquisition
    cache_dir = Path("./data/cache")
    acquisition = DataAcquisition(cache_dir=cache_dir)
    
    # List available sources
    print("Available data sources:")
    for source in acquisition.list_available_sources():
        config = acquisition.get_source_info(source)
        print(f"  - {source}: {config.source} ({config.format})")
    
    print("\n" + "="*60)
    
    # Example 1: Single dataset acquisition
    print("1. Acquiring Celestrak TLE data...")
    try:
        gps_data = await acquisition.acquire_dataset(
            'celestrak_tle',
            params={'catalog': 'gps-ops.txt'}
        )
        print(f"   ✓ GPS TLE data: {len(gps_data)} satellites")
        print(f"   Columns: {gps_data.columns}")
        print(f"   Sample data:")
        print(gps_data.head(3))
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n" + "-"*40)
    
    # Example 2: Multiple datasets concurrently  
    print("2. Acquiring multiple datasets...")
    acquisitions = [
        {
            'name': 'gps_satellites',
            'source': 'celestrak_tle', 
            'params': {'catalog': 'gps-ops.txt'}
        },
        {
            'name': 'space_debris', 
            'source': 'esa_space_debris',
            'params': {'limit': 50}
        },
        {
            'name': 'nasa_horizons',
            'source': 'nasa_jpl_horizons',
            'params': {
                'target': '399',
                'start_time': '2024-01-01',
                'stop_time': '2024-01-02'
            }
        }
    ]
    
    # Initialize datasets dictionary for error handling
    datasets = {}
    
    try:
        datasets = await acquisition.acquire_multiple(acquisitions, max_concurrent=2)
        
        for name, df in datasets.items():
            print(f"   ✓ {name}: {len(df)} records")
            
        # Show combined info
        total_records = sum(len(df) for df in datasets.values())
        print(f"   Total records across all datasets: {total_records}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        # Ensure datasets is defined even on error
        if not datasets:
            datasets = {}
    
    print("\n" + "="*60)
    
    # Example 3: Data processing
    print("3. Processing data with relativistic corrections...")
    
    processor = RelativisticDataProcessor()
    
    if 'gps_satellites' in datasets:
        processed_data = processor.process_dataframe(datasets['gps_satellites'])
        
        print(f"   Original columns: {len(datasets['gps_satellites'].columns)}")
        print(f"   Processed columns: {len(processed_data.columns)}")
        
        # Show new relativistic columns
        new_cols = set(processed_data.columns) - set(datasets['gps_satellites'].columns)
        if new_cols:
            print(f"   New relativistic features: {list(new_cols)}")
    
    print("\n" + "="*60)
    
    # Example 4: Creating PyTorch Geometric datasets
    print("4. Creating PyTorch Geometric datasets...")
    
    loader = SpaceDataLoader()
    
    try:
        # Print the available columns in the datasets to debug
        if 'gps_satellites' in datasets:
            print(f"DEBUG: GPS satellites columns: {datasets['gps_satellites'].columns}")
            print(f"DEBUG: GPS satellites shape: {datasets['gps_satellites'].shape}")
            print(f"DEBUG: First few rows:")
            print(datasets['gps_satellites'].head(3))
            
        # Create satellite constellation dataset (force rebuild to avoid cache issues)
        dataset = loader.load_satellite_constellation(
            root="./data/processed",
            data_source="gps",
            force_rebuild=False  # Set to True only if needed
        )
        
        info = loader.get_dataset_info(dataset)
        print(f"   ✓ Dataset created successfully:")
        for key, value in info.items():
            print(f"     {key}: {value}")
        
        # Create DataLoader
        dataloader = loader.create_data_loader(dataset, batch_size=4, shuffle=True)
        print(f"   ✓ DataLoader created with batch_size=4")
        
        # Show first batch
        first_batch = next(iter(dataloader))
        print(f"   ✓ First batch shape: {first_batch.x.shape}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    
    # Example 5: Visualization
    print("5. Creating visualizations...")
    
    visualizer = SpaceDataVisualizer()
    
    if 'gps_satellites' in datasets and len(datasets['gps_satellites']) > 0:
        try:
            # Check if we have the required columns for orbital elements
            df = datasets['gps_satellites']
            if all(col in df.columns for col in ['inclination_deg', 'eccentricity']):
                fig = visualizer.plot_orbital_elements(df)
                print(f"   ✓ Orbital elements plot created")
                
                # Save the plot
                output_dir = Path("./outputs")
                output_dir.mkdir(exist_ok=True)
                visualizer.save_figure(fig, output_dir / "orbital_elements.html")
                print(f"   ✓ Plot saved to {output_dir / 'orbital_elements.html'}")
            else:
                print(f"   ⚠️  Missing columns for orbital elements plot")
                print(f"      Available: {df.columns}")
                
        except Exception as e:
            print(f"   ❌ Visualization error: {e}")
    
    # Clean up
    await acquisition.close()
    print("\n✅ Example completed successfully!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
