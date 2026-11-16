from app.config import get_settings
from app.database.mongodb import get_database


def _col():
    return get_database()[get_settings().jobs_collection]


class MongoJobStore:
    """Stores jobs as plain dicts. It knows nothing about workflows."""

    async def save(self, job_id: str, doc: dict) -> None:
        body = {k: v for k, v in doc.items() if k != "job_id"}
        body["_id"] = job_id  # the job id is the primary key, so no extra index is needed
        await _col().replace_one({"_id": job_id}, body, upsert=True)

    async def get(self, job_id: str) -> dict | None:
        doc = await _col().find_one({"_id": job_id})
        if doc is None:
            return None
        doc["job_id"] = doc.pop("_id")
        return doc

    async def fail_orphans(self, finished_at: str) -> int:
        """Jobs left queued or running by a previous process can never finish."""
        result = await _col().update_many(
            {"status": {"$in": ["queued", "running"]}},
            {"$set": {"status": "failed", "failed_step": "server_restarted", "finished_at": finished_at}},
        )
        return result.modified_count