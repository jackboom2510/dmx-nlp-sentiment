#!/usr/bin/env python3
"""
Entry-point wrapper tại thư mục gốc cho scripts/auto_label_review.py.
Cho phép chạy trực tiếp:
    python auto_label_review.py [các tham số]
hoặc:
    python scripts/auto_label_review.py [các tham số]
"""

from scripts.auto_label_review import main

if __name__ == "__main__":
    main()
