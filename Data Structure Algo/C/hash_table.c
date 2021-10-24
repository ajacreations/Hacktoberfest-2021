#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SIZE 20

typedef struct data_item
{
    int data;
    int key;
} data_item_t;

data_item_t *hashArray[SIZE];

int
hashCode(int key)
{
    return key % SIZE;
}

data_item_t *
search(int key)
{
    // get the hash
    unsigned int hashIndex = hashCode(key);

    if (hashIndex >= SIZE) {
        return NULL;
    }

    // move in array until an empty
    while (hashArray[hashIndex] != NULL) {

        if (hashArray[hashIndex]->key == key) {
            return hashArray[hashIndex];
        }

        // go to next cell
        ++hashIndex;

        // wrap around the table
        hashIndex %= SIZE;
    }

    return NULL;
}

void
insert(void)
{
    int key, data;

    fputs("Please enter your key: ", stdout);
    scanf("%d", &key);
    fputs("Please enter your data for this key: ", stdout);
    scanf("%d", &data);

    data_item_t *item = (data_item_t *) malloc(sizeof(data_item_t));
    item->data = data;
    item->key = key;

    // get the hash
    unsigned int hashIndex = hashCode(key);
    if (hashIndex >= SIZE) {
        free(item);
        return;
    }

    // move in array until an empty or deleted cell
    while (hashArray[hashIndex] != NULL && hashArray[hashIndex]->key != -1) {
        // go to next cell
        ++hashIndex;

        // wrap around the table
        hashIndex %= SIZE;
    }

    hashArray[hashIndex] = item;
}

data_item_t *
delete(void)
{
    int data;
    fputs("Please enter key what you want to delete: ", stdout);
    scanf("%d", &data);

    data_item_t *item = search(data);
    if (item == NULL) {
        return NULL;
    }
    int key = item->key;

    // get the hash
    unsigned int hashIndex = hashCode(key);
    if (hashIndex >= SIZE) {
        return NULL;
    }

    // move in array until an empty
    while (hashArray[hashIndex] != NULL) {

        if (hashArray[hashIndex]->key == key) {
            data_item_t *temp = hashArray[hashIndex];

            // assign a dummy item at deleted position
            hashArray[hashIndex] = NULL;
            return temp;
        }

        // go to next cell
        ++hashIndex;

        // wrap around the table
        hashIndex %= SIZE;
    }

    return NULL;
}

void
display(void)
{
    for (int i = 0; i < SIZE; i++) {

        if (hashArray[i] != NULL) {
            printf(" (%d,%d)", hashArray[i]->key, hashArray[i]->data);
        } else {
            fputs(" ~~ ", stdout);
        }
    }

    fputs("\n", stdout);
}

int
main(void)
{
    int choice;

    puts("Hash Table Operations:");
    puts("\t1.Insert\n\t2.Delete\n\t3.Search\n\t4.Display\n\t5.Exit");
    do {
        fputs("Enter Your Choice: ", stdout);
        scanf("%d", &choice);
        switch (choice) {
            case 1: {
                insert();
                break;
            }
            case 2: {
                data_item_t *tmp = delete();
                if (tmp == NULL) {
                    fputs("Some error occuried!\n", stderr);
                } else {
                    puts("Deleted successfully!");
                }

                break;
            }
            case 3: {
                int key;
                fputs("Please enter key for search: ", stdout);
                scanf("%d", &key);

                data_item_t *tmp = search(key);
                if (tmp != NULL) {
                    printf("Element found: %d\n", tmp->data);
                } else {
                    printf("Element not found\n");
                }

                break;
            }
            case 4: {
                display();
                break;
            }
            case 5: {
                puts("Bye :)");
                break;
            }
            default: {
                puts("Enter a correct choice (1,2,3,4,5)");
                break;
            }
        }
    } while (choice != 5);

    return 0;
}
