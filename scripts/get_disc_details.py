from huggingface_hub import HfApi

api = HfApi()
disc = api.get_discussion_details(repo_id="Ravikishore710/polyglot-mansu-tiny-aya-fire", repo_type="model", discussion_num=2)
print("Title:", disc.title)
print("Author:", disc.author)
print("Status:", disc.status)
print("Num events:", len(disc.events))
for idx, ev in enumerate(disc.events):
    print(f"Event {idx}: type={ev.type}, id={getattr(ev, 'id', None)}")
