import asyncio
from app.services.recommendations import get_engine

async def main():
    print("Testing recommendation engine...")
    engine = get_engine()
    
    # Initialize engine (will fallback to CSV because no DB session is provided)
    print("Initializing recommendation engine...")
    await engine.initialize()
    print("Engine initialization completed successfully.")
    
    user_id = "554ae8abb4ec7fd017001796" # one of the top active users from CSV
    print(f"Testing recommendations for user: {user_id}")
    
    # 1. Test Service Recommendations
    print("\n--- 1. Service Recommendations ---")
    services_res = await engine.recommend(user_id=user_id, n=5, is_dataset=False)
    print(f"Algorithm used: {services_res.algorithm_used}")
    print("Recommendations:")
    for rec in services_res.recommendations:
        rec_dict = rec.to_dict(is_dataset=False)
        print(f"  Service ID: {rec_dict['service_id']}, Score: {rec_dict['score']}, Reason: {rec_dict['reason']}")
        assert rec_dict['service_id'] < 1000000, f"Error: Dataset {rec_dict['service_id']} found in service recommendations!"
    
    # 2. Test Dataset Recommendations
    print("\n--- 2. Dataset Recommendations ---")
    datasets_res = await engine.recommend(user_id=user_id, n=5, is_dataset=True)
    print(f"Algorithm used: {datasets_res.algorithm_used}")
    print("Recommendations:")
    for rec in datasets_res.recommendations:
        rec_dict = rec.to_dict(is_dataset=True)
        print(f"  Dataset ID: {rec_dict['dataset_id']}, Score: {rec_dict['score']}, Reason: {rec_dict['reason']}")
        # Make sure original dataset ID is returned (without the 1M offset)
        assert rec_dict['dataset_id'] < 1000000, f"Error: Dataset ID {rec_dict['dataset_id']} should have the offset subtracted!"
        
    # 3. Test Service Recommendations with dataset_id filter (with empty dataset connection mapping under CSV mode)
    print("\n--- 3. Service Recommendations with dataset_id filter ---")
    services_filtered = await engine.recommend(user_id=user_id, n=5, is_dataset=False, dataset_id=42)
    print(f"Filtered recommendations count: {len(services_filtered.recommendations)}")
    
    print("\nAll unit tests passed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
