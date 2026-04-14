import argparse

def main() -> None:
    parser = argparse.ArgumentParser(description="Continuoso CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Add command
    add_parser = subparsers.add_parser("add", help="Add a new item (placeholder)")
    add_parser.add_argument("item_name", type=str, help="Name of the item to add")

    # List command
    list_parser = subparsers.add_parser("list", help="List all items (placeholder)")

    # Complete command
    complete_parser = subparsers.add_parser("complete", help="Mark an item as complete (placeholder)")
    complete_parser.add_argument("item_id", type=int, help="ID of the item to complete")

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete an item (placeholder)")
    delete_parser.add_argument("item_id", type=int, help="ID of the item to delete")

    args = parser.parse_args()

    if args.command == "add":
        print(f"Placeholder: Adding item '{args.item_name}'")
    elif args.command == "list":
        print("Placeholder: Listing items")
    elif args.command == "complete":
        print(f"Placeholder: Completing item with ID '{args.item_id}'")
    elif args.command == "delete":
        print(f"Placeholder: Deleting item with ID '{args.item_id}'")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
