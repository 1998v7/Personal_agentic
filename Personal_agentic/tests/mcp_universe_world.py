"""对拍用的"假世界":按参数确定性地生成 GitHub / Google Maps / Notion 的 MCP 返回,上游与 IMAct 两边共用。返回的是 MCP 结果的 JSON 形式。"""
import hashlib
import json


def h(*parts) -> int:
    return int(hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest(), 16)


def text(payload) -> dict:
    return {"content": [{"type": "text", "text": payload if isinstance(payload, str) else json.dumps(payload)}], "isError": False}


def error(msg) -> dict:
    return {"content": [{"type": "text", "text": msg}], "isError": True}


CITIES = ["Kuala Lumpur", "Johor Bahru", "Melaka", "Seremban", "Singapore", "Penang"]


def normalize(server: str, tool: str, args: dict) -> dict:
    """Notion 搜索:上游把请求体序列化成字符串塞进 query,IMAct 传纯字符串 + 单独的 filter。两种写法在假世界里是同一个请求。"""
    if server == "notion" and tool == "API-post-search":
        q = args["query"]
        if isinstance(q, str):
            try:
                body = json.loads(q)
                if isinstance(body, dict):
                    q = body["query"]
            except ValueError:
                pass
        return {"query": q}
    return args


def call(server: str, tool: str, args: dict) -> dict:
    args = normalize(server, tool, args) if server == "notion" and tool == "API-post-search" else args
    if server == "github" and tool == "get_file_contents" and "branch" in args:
        # 新版 server 的参数名是 branch,旧版是 ref:IMAct 两个都传,上游只传 ref。在假世界里是同一个请求。
        args = {k: v for k, v in args.items() if k != "branch"}
        args.setdefault("ref", args.pop("ref", None))
    n = h(server, tool, args)
    if server == "github":
        owner = args.get("owner", "testuser")
        if tool == "search_repositories":
            q = args["query"]
            if n % 5 == 0:
                return text({"total_count": 0, "items": []})
            name = q.split("/")[-1].split(" ")[0].replace("repo:", "") or "repo"
            return text({"total_count": 1, "items": [{"full_name": q if "/" in q and " " not in q else f"testuser/{name}", "name": name, "owner": {"login": "testuser"}}]})
        if tool == "get_file_contents":
            if n % 7 == 0:
                return error("not found")
            body = f"content of {args.get('path')}@{args.get('ref')}" if n % 2 else "# Travel Planner App\n\nA comprehensive travel planning application."
            return {"content": [{"type": "text", "text": "successfully downloaded text file"},
                                {"type": "resource", "resource": {"uri": f"repo://{owner}/{args.get('repo')}/{args.get('path')}", "mimeType": "text/plain", "text": body}}], "isError": False}
        if tool == "list_branches":
            if n % 6 == 0:
                return error("boom")
            return text([{"name": "main"}, {"name": "feature-maps"}, {"name": "feature-itinerary"}][: 1 + n % 3])
        if tool == "list_pull_requests":
            return text([] if n % 3 == 0 else [{"title": "Add basic Express server setup", "body": "This PR implements the initial Express server configuration with a basic route handler for the homepage."}, {"title": "other", "body": None}])
        if tool == "list_issues":
            if args.get("page", 1) > 1:
                return text([])
            return text([{"number": i, "title": f"Issue {i}" if i % 2 else "Bug", "labels": [{"name": "bug"}, {"name": f"l{i}"}]} for i in range(1, 1 + n % 5)])
        if tool == "get_issue_comments":
            return text([{"body": "thanks"}, {"body": f"comment {args.get('issue_number')}"}])
    if server == "google-maps":
        if tool == "maps_search_places":
            q = args["query"]
            k = 1 + n % 3
            if n % 11 == 0:
                return text({"places": []})
            return text({"places": [{"name": f"{q} #{i}", "formatted_address": f"{i} Jalan {CITIES[(n + i) % len(CITIES)]}, Malaysia", "place_id": f"pid{n % 1000}_{i}",
                                     "location": {"lat": 3.0 + (n % 100) / 100 + i, "lng": 101.0 + (n % 50) / 100}, "rating": 3 + (n + i) % 3 * 0.5, "types": ["restaurant", "point_of_interest"]}
                                    for i in range(k)]})
        if tool == "maps_geocode":
            c = CITIES[n % len(CITIES)]
            return text({"location": {"lat": 3.1 + (n % 10) / 10, "lng": 101.6}, "formatted_address": f"{args['address']}, {c}, Malaysia", "place_id": f"g{n % 1000}",
                         "address_components": [{"long_name": c, "short_name": c, "types": ["locality"]}, {"long_name": "Malaysia", "short_name": "MY", "types": ["country"]}]})
        if tool == "maps_place_details":
            return text({"name": f"place {args['place_id']}", "formatted_address": f"x Rd, {CITIES[n % len(CITIES)]}, Malaysia", "rating": 3 + n % 3 * 0.5, "place_id": args["place_id"],
                         "location": {"lat": 3.2, "lng": 101.7}, "types": ["park"], "address_components": [{"long_name": CITIES[n % len(CITIES)], "types": ["locality"]}]})
        if tool == "maps_distance_matrix":
            return text({"origin_addresses": args["origins"], "destination_addresses": args["destinations"],
                         "results": [{"elements": [{"status": "OK", "duration": {"text": f"{1 + n % 5} hours", "value": 3600 * (1 + n % 5)}, "distance": {"text": f"{100 + n % 300} km", "value": 1000 * (100 + n % 300)}}]}]})
        if tool == "maps_elevation":
            return text({"results": [{"elevation": float(n % 900), "location": args["locations"][0], "resolution": 10.0}]})
    if server == "notion":
        if tool == "API-post-search":
            q = args["query"]
            if n % 4 == 0:
                return text({"results": []})
            return text({"results": [{"id": f"page{n % 100}", "properties": {"title": {"title": [{"text": {"content": q}}]}}}]})
        if tool == "API-get-block-children":
            blocks = [{"type": "paragraph", "paragraph": {"rich_text": [{"plain_text": f"line {i}"}]}} for i in range(n % 4)]
            return text({"results": blocks})
    return error(f"unknown tool {server}.{tool}")
