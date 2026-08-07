#!/bin/bash
curl -s http://127.0.0.1:8080/health && echo
curl -s http://127.0.0.1:8080/v1/models && echo
