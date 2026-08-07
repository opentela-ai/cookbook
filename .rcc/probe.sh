#!/bin/bash
cat < /dev/null > /dev/tcp/140.238.223.116/43905 && echo "140.238.223.116:43905 reachable" || echo "140.238.223.116:43905 NOT reachable"
