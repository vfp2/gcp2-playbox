#!/bin/bash
# Monitor MQTT metrics in real-time

echo "Monitoring MQTT activity (press Ctrl+C to stop)..."
echo "================================================"
echo ""

while true; do
    clear
    echo "GCP2 MQTT Monitor - $(date)"
    echo "================================================"

    # Connections
    connects=$(aws cloudwatch get-metric-statistics \
        --namespace AWS/IoT \
        --metric-name Connect.Success \
        --dimensions Name=Protocol,Value=MQTT \
        --start-time $(date -u -d '2 minutes ago' +%Y-%m-%dT%H:%M:%S) \
        --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
        --period 120 \
        --statistics Sum \
        --profile vfp2 \
        --region us-east-2 \
        --query 'Datapoints[0].Sum' \
        --output text 2>/dev/null)

    # Publishes
    publishes=$(aws cloudwatch get-metric-statistics \
        --namespace AWS/IoT \
        --metric-name PublishIn.Success \
        --dimensions Name=Protocol,Value=MQTT \
        --start-time $(date -u -d '2 minutes ago' +%Y-%m-%dT%H:%M:%S) \
        --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
        --period 120 \
        --statistics Sum \
        --profile vfp2 \
        --region us-east-2 \
        --query 'Datapoints[0].Sum' \
        --output text 2>/dev/null)

    echo "Last 2 minutes:"
    echo "  Connections: ${connects:-0}"
    echo "  Messages Published: ${publishes:-0}"
    echo ""
    echo "Keep MQTT test client open to see messages when they arrive!"
    echo "AWS Console: https://us-east-2.console.aws.amazon.com/iot/home?region=us-east-2#/test"

    sleep 10
done
