set -o errexit
set -o pipefail
set -o nounset

echo "======================================================="
echo "python tests/test_folder_summary.py $@"
echo "======================================================="
python tests/test_folder_summary.py

echo "======================================================="
echo "python tests/test_read_file.py $@"
echo "======================================================="
python tests/test_read_file.py

# echo "======================================================="
# echo "python tests/test_wikipedia_search.py $@"
# echo "======================================================="
# python tests/test_wikipedia_search.py

echo "======================================================="
echo "python tests/test_paper_search.py  $@"
echo "======================================================="
python tests/test_paper_search.py
