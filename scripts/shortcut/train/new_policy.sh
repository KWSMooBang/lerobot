#!/usr/bin/env bash
# Copy the sc_template plugin into a new policy package and a matching preset.
#
#   scripts/shortcut/train/new_policy.sh my_idea
#     -> scripts/shortcut/policies/lerobot_policy_my_idea/  (policy type "my_idea", class MyIdeaPolicy)
#     -> scripts/shortcut/train/presets/my_idea_libero_all.yaml
#   sc-train --preset my_idea_libero_all --run-id tb-my-idea-a
set -euo pipefail

name=${1:-}
if ! [[ $name =~ ^[a-z][a-z0-9_]*$ ]]; then
  echo "usage: $0 <name>   (lowercase snake_case, e.g. lang_film)" >&2
  exit 1
fi
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
POL=$HERE/../policies
src=$POL/lerobot_policy_sc_template
dst=$POL/lerobot_policy_$name
preset=$HERE/presets/${name}_libero_all.yaml
[ -e "$dst" ] && { echo "exists: $dst" >&2; exit 1; }
[ -e "$preset" ] && { echo "exists: $preset" >&2; exit 1; }

camel=$(echo "$name" | awk -F_ '{for (i = 1; i <= NF; i++) printf "%s", toupper(substr($i, 1, 1)) substr($i, 2)}')

cp -r "$src" "$dst"
find "$dst" -name __pycache__ -prune -exec rm -rf {} +
find "$dst" -name '*.egg-info' -prune -exec rm -rf {} +
mv "$dst/src/lerobot_policy_sc_template" "$dst/src/lerobot_policy_$name"
for f in "$dst"/src/lerobot_policy_"$name"/*_sc_template.py; do
  mv "$f" "${f/_sc_template.py/_$name.py}"
done
grep -rl --null -e sc_template -e ScTemplate "$dst" | xargs -0 sed -i -e "s/sc_template/$name/g" -e "s/ScTemplate/$camel/g"

sed -e "s/sc_template/$name/g" -e "s/^description: .*/description: $name (copied from sc_template)/" \
  "$HERE/presets/sc_template_libero_all.yaml" > "$preset"

echo "created $dst"
echo "created $preset"
echo "next: edit $dst/src/lerobot_policy_$name/modeling_$name.py, then"
echo "  scripts/shortcut/train/sc-train --preset ${name}_libero_all --run-id tb-$name-a --steps 2000"
