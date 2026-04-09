"""A simplified data processing workflow demonstrating core Mistral Workflows concepts."""

from typing import List, Dict, Any
from pydantic import BaseModel
import mistralai.workflows as workflows
import random
import asyncio


class DataProcessingInput(BaseModel):
    data_source: str = "sample"
    batch_size: int = 5
    fail_probability: float = 0.0  # For simulating failures


class ProcessedItem(BaseModel):
    id: str
    value: float
    processed: bool
    error: str | None = None


@workflows.activity(name="fetch-data")
async def fetch_data(source: str, batch_size: int) -> List[Dict[str, Any]]:
    """Simulate fetching data from an external source."""
    print(f"Fetching {batch_size} items from {source}...")
    
    # Simulate API call delay
    await asyncio.sleep(1)
    
    # Generate sample data
    return [
        {"id": f"item_{i}", "value": random.uniform(1.0, 100.0)}
        for i in range(batch_size)
    ]


@workflows.activity(name="process-item")
async def process_item(item: Dict[str, Any], fail_probability: float) -> ProcessedItem:
    """Process a single item with potential failure simulation."""
    print(f"Processing item {item['id']}...")
    
    # Simulate processing time
    await asyncio.sleep(0.5)
    
    # Simulate potential failure
    if random.random() < fail_probability:
        raise ValueError(f"Simulated processing failure for {item['id']}")
    
    # Successful processing
    processed_value = item["value"] * 1.1  # 10% increase as processing
    return ProcessedItem(
        id=item["id"],
        value=processed_value,
        processed=True,
        error=None
    )


@workflows.activity(name="handle-error")
async def handle_error(item: Dict[str, Any], error: str) -> ProcessedItem:
    """Handle errors for failed items."""
    print(f"Handling error for item {item['id']}: {error}")
    
    await asyncio.sleep(0.3)
    
    return ProcessedItem(
        id=item["id"],
        value=item["value"],  # Return original value
        processed=False,
        error=f"Processing failed: {error}"
    )


@workflows.workflow.define(
    name="simple-data-processor",
    workflow_display_name="Simple Data Processor",
    workflow_description="A simple workflow that fetches and processes data with error handling.",
)
class SimpleDataProcessorWorkflow:
    @workflows.workflow.entrypoint
    async def run(self, input: DataProcessingInput) -> List[ProcessedItem]:
        """Main workflow entry point."""
        
        # Step 1: Fetch data
        raw_data = await fetch_data(input.data_source, input.batch_size)
        
        # Step 2: Process items sequentially with error handling
        processed_items = []
        
        for item in raw_data:
            try:
                # Process each item
                processed_item = await process_item(item, input.fail_probability)
                processed_items.append(processed_item)
                
            except Exception as e:
                # Handle errors gracefully
                error_item = await handle_error(item, str(e))
                processed_items.append(error_item)
        
        # Step 3: Return results
        return processed_items