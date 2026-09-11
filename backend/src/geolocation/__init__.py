"""Geolocation / geotagging engine.

Converts a detection's pixel position in a side-scan sonar frame into a
real-world latitude/longitude, using the tow vehicle's navigation fix for
that frame plus the across-track (cross-swath) pixel offset. Per the
challenge brief's "Anomalous Reporting & Geotagging Engine" requirement:
reads sonar metadata (nav headers / a coordinate sidecar) and outputs
lat/lon for every detected hazard.

There was previously no code here at all (only a static robo_output.geojson
output artifact, confirmed empty on inspection). Everything in this package
is new.
"""
