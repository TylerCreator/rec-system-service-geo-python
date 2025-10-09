"""
Update service - orchestrates all system update operations
"""
import asyncio
import json
import os
from datetime import datetime
from typing import List, Dict, Any
from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import Call, Service, User, UserService
from app.core.database import AsyncSessionLocal
from app.core.config import settings
from app.services import calls_service, services_service, datasets_service, compositions_service


async def update_all(db: AsyncSession = None) -> List[str]:
    """
    Update all data: calls, datasets, services, compositions
    """
    results = []
    
    # Create DB session if not provided
    if db is None:
        async with AsyncSessionLocal() as db:
            return await _update_all_impl(db)
    else:
        return await _update_all_impl(db)


async def _update_all_impl(db: AsyncSession) -> List[str]:
    """Implementation of update_all"""
    results = []
    
    try:
        print("Starting updateAll process...")
        
        # 1. Update calls
        try:
            print("Step 1: Updating calls...")
            await calls_service.update_calls(db)
            results.append("✅ Calls updated successfully")
        except Exception as e:
            print(f"Error updating calls: {e}")
            results.append(f"❌ Calls update failed: {str(e)}")
        
        # 2. Update datasets
        try:
            print("Step 2: Updating datasets...")
            await datasets_service.update_datasets(db)
            results.append("✅ Datasets updated successfully")
        except Exception as e:
            print(f"Error updating datasets: {e}")
            results.append(f"❌ Datasets update failed: {str(e)}")
        
        # 3. Update services
        try:
            print("Step 3: Updating services...")
            await services_service.update_services(db)
            results.append("✅ Services updated successfully")
        except Exception as e:
            print(f"Error updating services: {e}")
            results.append(f"❌ Services update failed: {str(e)}")
        
        # 4. Recover compositions
        try:
            print("Step 4: Recovering compositions...")
            await compositions_service.recover(db)
            results.append("✅ Compositions recovered successfully")
        except Exception as e:
            print(f"Error recovering compositions: {e}")
            results.append(f"❌ Compositions recovery failed: {str(e)}")
        
        print("UpdateAll process completed")
        return results
        
    except Exception as e:
        print(f"Fatal error in updateAll: {e}")
        results.append(f"💥 Fatal error: {str(e)}")
        raise


async def update_statistics_internal(db: AsyncSession) -> str:
    """
    Internal function to update user-service statistics
    """
    print("Updating statistics internally...")
    
    try:
        # Use raw SQL for aggregation
        query = text("""
            SELECT owner, mid, COUNT(*) as call_count 
            FROM "Calls" 
            WHERE owner IS NOT NULL AND mid IS NOT NULL 
            GROUP BY owner, mid
        """)
        
        result = await db.execute(query)
        call_stats = result.fetchall()
        
        print(f"Processing {len(call_stats)} user-service combinations")
        
        if not call_stats:
            print("No call statistics found")
            return "No call statistics to process"
        
        # Get unique owners and service IDs
        unique_owners = list(set(stat[0] for stat in call_stats))
        unique_service_ids = list(set(stat[1] for stat in call_stats))
        
        print(f"Found {len(unique_owners)} unique users and {len(unique_service_ids)} unique services")
        
        # Check existing services
        result = await db.execute(
            select(Service.id).where(Service.id.in_(unique_service_ids))
        )
        existing_service_ids = set(row[0] for row in result.fetchall())
        print(f"Found {len(existing_service_ids)} existing services in database")
        
        # Create users
        for owner in unique_owners:
            result = await db.execute(select(User).where(User.id == owner))
            existing_user = result.scalar_one_or_none()
            
            if not existing_user:
                user = User(id=owner)
                db.add(user)
        
        await db.commit()
        print("Users created/found successfully")
        
        # Update user-service statistics
        processed_count = 0
        skipped_count = 0
        
        for owner, mid, call_count in call_stats:
            # Check if service exists
            if mid not in existing_service_ids:
                print(f"Service with id {mid} not found, skipping...")
                skipped_count += 1
                continue
            
            # Find or create UserService
            result = await db.execute(
                select(UserService).where(
                    UserService.user_id == owner,
                    UserService.service_id == mid
                )
            )
            user_service = result.scalar_one_or_none()
            
            if user_service:
                # Update existing
                if user_service.number_of_calls != call_count:
                    user_service.number_of_calls = call_count
                    print(f"Updated calls count for user {owner} and service {mid}: {call_count}")
            else:
                # Create new
                user_service = UserService(
                    user_id=owner,
                    service_id=mid,
                    number_of_calls=call_count
                )
                db.add(user_service)
            
            processed_count += 1
        
        await db.commit()
        print("Statistics updated successfully")
        
        result_msg = f"Processed {processed_count} user-service combinations for {len(unique_owners)} users. Skipped {skipped_count} combinations due to missing services."
        print(result_msg)
        return result_msg
        
    except Exception as e:
        await db.rollback()
        print(f"Error in updateStatisticsInternal: {e}")
        raise Exception(f"Failed to update statistics: {str(e)}")


async def update_statistics(db: AsyncSession) -> str:
    """Update statistics (wrapper for HTTP endpoint)"""
    print("Starting statistics update...")
    try:
        result = await update_statistics_internal(db)
        print("Statistics update completed successfully")
        return result
    except Exception as e:
        print(f"Error updating statistics: {e}")
        raise


async def update_recomendations() -> Dict[str, Any]:
    """
    Update recommendations using KNN Python script
    """
    try:
        print("Starting recommendations update...")
        
        # Ensure files exist
        if not os.path.exists(settings.CSV_FILE_PATH):
            raise Exception(f"CSV file not found: {settings.CSV_FILE_PATH}")
        
        if not os.path.exists(settings.KNN_SCRIPT_PATH):
            raise Exception(f"KNN script not found: {settings.KNN_SCRIPT_PATH}")
        
        # Prepare recommendations file
        recommendations_file = settings.RECOMMENDATIONS_FILE_PATH
        temp_file = "recomendations_temp.json"
        
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump({"prediction": {}}, f)
            print("Temporary file created successfully")
            
            if os.path.exists(recommendations_file):
                try:
                    os.remove(recommendations_file)
                    print("Old recommendations file removed")
                except Exception as e:
                    print(f"Could not remove old file: {e}")
            
            os.rename(temp_file, recommendations_file)
            print("Recommendations file prepared successfully")
            
        except Exception as e:
            print(f"Could not create recommendations file: {e}")
        
        # Run Python KNN script
        proc = await asyncio.create_subprocess_exec(
            "python3",
            settings.KNN_SCRIPT_PATH,
            settings.CSV_FILE_PATH,
            "",  # No specific user_id for full update
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60.0)
        
        if proc.returncode != 0:
            print(f"Python process failed: {stderr.decode()}")
            return {
                "success": False,
                "error": "Failed to update recommendations",
                "message": f"Python process exited with code {proc.returncode}",
                "details": stderr.decode()[:1000],
                "timestamp": datetime.now().isoformat()
            }
        
        # Parse output
        try:
            output_data = stdout.decode()
            if output_data.strip():
                result = json.loads(output_data)
                print("Recommendations updated successfully")
            else:
                # Read from file
                with open(recommendations_file, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                print("Recommendations read from file successfully")
            
            return {
                "success": True,
                "message": "Recommendations updated successfully",
                "data": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as parse_error:
            print(f"Failed to parse recommendations data: {parse_error}")
            return {
                "success": False,
                "error": "Failed to parse recommendations data",
                "message": str(parse_error),
                "timestamp": datetime.now().isoformat()
            }
        
    except asyncio.TimeoutError:
        print("Python process timeout")
        return {
            "success": False,
            "error": "Recommendation update timeout",
            "message": "Process took too long to complete",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"Error in updateRecomendations: {e}")
        return {
            "success": False,
            "error": "Internal server error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }


async def run_full_update() -> Dict[str, Any]:
    """
    Full system update (for cron job and manual trigger)
    Runs all update operations in sequence
    """
    print(f"🕐 Starting full system update at: {datetime.now().isoformat()}")
    start_time = datetime.now()
    results = []
    
    try:
        # Use new database session
        async with AsyncSessionLocal() as db:
            # 1. Update all data
            print("📊 Step 1: Updating all data...")
            try:
                main_results = await _update_all_impl(db)
                results.extend(main_results)
                
                has_main_errors = any('❌' in r for r in main_results)
                if has_main_errors:
                    results.append("❌ updateAll: COMPLETED WITH ERRORS")
                    print("⚠️ updateAll completed with some errors")
                else:
                    results.append("✅ updateAll: SUCCESS")
                    print("✅ updateAll completed successfully")
            except Exception as e:
                results.append(f"❌ updateAll: FAILED - {str(e)}")
                print(f"❌ updateAll failed: {e}")
            
            # 2. Dump CSV
            print("📄 Step 2: Dumping CSV...")
            try:
                await calls_service.dump_csv(db)
                results.append("✅ dumpCsv: SUCCESS")
                print("✅ dumpCsv completed successfully")
            except Exception as e:
                results.append(f"❌ dumpCsv: FAILED - {str(e)}")
                print(f"❌ dumpCsv failed: {e}")
            
            # 3. Update statistics
            print("📈 Step 3: Updating statistics...")
            try:
                await update_statistics_internal(db)
                results.append("✅ updateStatics: SUCCESS")
                print("✅ updateStatics completed successfully")
            except Exception as e:
                results.append(f"❌ updateStatics: FAILED - {str(e)}")
                print(f"❌ updateStatics failed: {e}")
        
        # 4. Update recommendations (outside DB session)
        print("🤖 Step 4: Updating recommendations...")
        try:
            rec_result = await update_recomendations()
            if rec_result.get("success"):
                results.append("✅ updateRecomendations: SUCCESS")
                print("✅ updateRecomendations completed successfully")
            else:
                results.append(f"❌ updateRecomendations: FAILED - {rec_result.get('message')}")
                print(f"❌ updateRecomendations failed")
        except Exception as e:
            results.append(f"❌ updateRecomendations: FAILED - {str(e)}")
            print(f"❌ updateRecomendations failed: {e}")
        
    except Exception as critical_error:
        print(f"💥 Critical error in full update: {critical_error}")
        results.append(f"💥 CRITICAL ERROR: {str(critical_error)}")
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60  # in minutes
    
    print("🏁 Full system update completed")
    print(f"⏱️  Total execution time: {duration:.2f} minutes")
    print("📋 Results summary:", results)
    print("=" * 80)
    
    has_errors = any('❌' in r or '💥' in r for r in results)
    has_critical = any('💥' in r for r in results)
    
    return {
        "success": not has_critical,
        "message": "Full system update completed",
        "executionTime": f"{duration:.2f} minutes",
        "results": results,
        "hasErrors": has_errors,
        "timestamp": datetime.now().isoformat()
    }

