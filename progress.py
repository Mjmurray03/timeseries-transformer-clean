import re
from pathlib import Path

def check_progress():
    completed = []
    remaining = []
    
    for spec_dir in Path('.kiro/specs').iterdir():
        if spec_dir.is_dir():
            tasks_file = spec_dir / 'tasks.md'
            if tasks_file.exists():
                content = tasks_file.read_text()
                # Find all tasks
                all_tasks = re.findall(r'(TASK-\w+)', content)
                
                # Check which are implemented (you'll mark these)
                for task in all_tasks:
                    if f"[x] **{task}" in content:
                        completed.append(task)
                    else:
                        remaining.append(task)
    
    print(f"✅ Completed: {len(completed)} tasks")
    print(f"⏳ Remaining: {len(remaining)} tasks")
    print(f"📊 Progress: {len(completed)/(len(completed)+len(remaining))*100:.1f}%")
    
    return completed, remaining

if __name__ == "__main__":
    check_progress()