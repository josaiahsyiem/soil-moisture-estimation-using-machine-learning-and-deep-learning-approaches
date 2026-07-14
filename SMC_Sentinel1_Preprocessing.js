/*=============================================================================
  Sentinel-1 GRD Dual-Polarimetric Descriptors for Soil Moisture Retrieval
  Google Earth Engine Preprocessing Script

  Author: Josaiah Syiem
  Institution: Centre of Studies in Resources Engineering (CSRE), IIT Bombay
  Supervisor: Prof. Y.S. Rao

  Description:
    Computes dual-polarimetric descriptors from Sentinel-1 GRD data over the
    Texas Soil Observation Network (TxSON) study area. The following
    descriptors are computed per acquisition and extracted at TxSON station
    locations for use in the soil moisture retrieval ML pipeline:

      - DpRVIc : Dual-pol Radar Vegetation Index (corrected for GRD)
                 q(q+3)/(q+1)^2 where q = C22/C11 (VH_lin/VV_lin)
                 Range: 0 (bare soil) → 1 (dense vegetation)
      - Hc     : Pseudo scattering entropy
      - Theta_c: Pseudo scattering type parameter
      - mc     : Co-pol purity parameter
      - VV, VH : Mean backscatter coefficients (linear scale)
      - inc    : Radar incidence angle

  Methodology reference:
    Bhogapurapu et al. (2022). Field-scale soil moisture estimation using
    Sentinel-1 GRD SAR data. Advances in Space Research, 70(10), 3845-3858.
    https://doi.org/10.1016/j.asr.2022.03.019

  Station coordinates:
    Caldwell, T., Bongiovanni, T. et al. (2019). Texas Soil Observation
    Network (TxSON). Texas Data Repository, V5.
    https://doi.org/10.18738/T8/JJ16CF
=============================================================================*/


// ─── 1. STUDY AREA AND STATION LOCATIONS ─────────────────────────────────────

var extent = ee.Geometry.Rectangle([-99.1, 30.0, -98.5, 30.5]);

// TxSON in-situ station locations (38 stations, 2015-2019)
var sample_pts = ee.FeatureCollection([
  ee.Feature(ee.Geometry.Point([-98.805859, 30.437585]), {label: '2_1'}),
  ee.Feature(ee.Geometry.Point([-98.746156, 30.289700]), {label: '2_2'}),
  ee.Feature(ee.Geometry.Point([-98.806510, 30.428332]), {label: '2_3'}),
  ee.Feature(ee.Geometry.Point([-98.779219, 30.429788]), {label: '2_4'}),
  ee.Feature(ee.Geometry.Point([-98.770115, 30.238091]), {label: '2_5'}),
  ee.Feature(ee.Geometry.Point([-98.703682, 30.238341]), {label: '2_6'}),
  ee.Feature(ee.Geometry.Point([-98.708413, 30.231818]), {label: '2_7'}),
  ee.Feature(ee.Geometry.Point([-98.686356, 30.283366]), {label: '2_8'}),
  ee.Feature(ee.Geometry.Point([-98.813330, 30.431926]), {label: '2_9'}),
  ee.Feature(ee.Geometry.Point([-98.770263, 30.307190]), {label: '2_10'}),
  ee.Feature(ee.Geometry.Point([-98.768479, 30.231682]), {label: '2_11'}),
  ee.Feature(ee.Geometry.Point([-98.833061, 30.210836]), {label: '2_12'}),
  ee.Feature(ee.Geometry.Point([-98.858269, 30.432736]), {label: '2_13'}),
  ee.Feature(ee.Geometry.Point([-98.802515, 30.415149]), {label: '2_14'}),
  ee.Feature(ee.Geometry.Point([-98.706910, 30.250071]), {label: '2_15'}),
  ee.Feature(ee.Geometry.Point([-98.741693, 30.283589]), {label: '2_16'}),
  ee.Feature(ee.Geometry.Point([-98.726830, 30.275365]), {label: '2_17'}),
  ee.Feature(ee.Geometry.Point([-98.698761, 30.245561]), {label: '2_18'}),
  ee.Feature(ee.Geometry.Point([-98.854222, 30.417532]), {label: '2_19'}),
  ee.Feature(ee.Geometry.Point([-98.783865, 30.421815]), {label: '2_21'}),
  ee.Feature(ee.Geometry.Point([-98.860405, 30.431540]), {label: '2_22'}),
  ee.Feature(ee.Geometry.Point([-98.654270, 30.202172]), {label: '2_23'}),
  ee.Feature(ee.Geometry.Point([-98.699027, 30.253402]), {label: '2_24'}),
  ee.Feature(ee.Geometry.Point([-98.699498, 30.249212]), {label: '2_25'}),
  ee.Feature(ee.Geometry.Point([-98.804610, 30.419318]), {label: '2_26'}),
  ee.Feature(ee.Geometry.Point([-98.847983, 30.448705]), {label: '2_29'}),
  ee.Feature(ee.Geometry.Point([-98.803321, 30.420466]), {label: '10_1'}),
  ee.Feature(ee.Geometry.Point([-98.705880, 30.245355]), {label: '10_2'}),
  ee.Feature(ee.Geometry.Point([-98.724202, 30.275773]), {label: '10_3'}),
  ee.Feature(ee.Geometry.Point([-98.610477, 30.398906]), {label: '10_5'}),
  ee.Feature(ee.Geometry.Point([-98.842698, 30.442127]), {label: '10_6'}),
  ee.Feature(ee.Geometry.Point([-98.614223, 30.430671]), {label: 'L_1'}),
  ee.Feature(ee.Geometry.Point([-98.851850, 30.420590]), {label: 'L_2'}),
  ee.Feature(ee.Geometry.Point([-98.613590, 30.171280]), {label: 'L_3'}),
  ee.Feature(ee.Geometry.Point([-98.868545, 30.342471]), {label: 'L_4'}),
  ee.Feature(ee.Geometry.Point([-98.612308, 30.330007]), {label: 'L_5'}),
  ee.Feature(ee.Geometry.Point([-98.632870, 30.240745]), {label: 'L_6'}),
  ee.Feature(ee.Geometry.Point([-98.949763, 30.166343]), {label: 'L_7'})
]);


// ─── 2. LOAD AND FILTER SENTINEL-1 GRD ───────────────────────────────────────

var ref_start = ee.Date('2015-01-01');
var ref_end   = ee.Date('2019-12-31');

var window_size = 2.5; // spatial averaging window (pixels)

var S1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterDate(ref_start, ref_end)
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
  .select('VV', 'VH', 'angle')
  .filterBounds(extent)
  .sort('system:time_start', false);


// ─── 3. BUILD DAILY MOSAICS ───────────────────────────────────────────────────

// Mosaic all acquisitions from the same day into a single image.
// This handles overlapping swaths from multiple relative orbits.

var diff  = ref_end.difference(ref_start, 'day');
var range = ee.List.sequence(0, diff.subtract(1)).map(function(day) {
  return ref_start.advance(day, 'day');
});

var day_mosaics = function(date, newlist) {
  date    = ee.Date(date);
  newlist = ee.List(newlist);
  var filtered = S1.filterDate(date, date.advance(1, 'day'));
  var image = ee.Image(filtered.mosaic())
    .set('system:time_start', filtered.first().get('system:time_start'))
    .set('system:index',      filtered.first().get('system:index'))
    .set('system:footprint',  filtered.first().get('system:footprint'));
  return ee.List(
    ee.Algorithms.If(filtered.size(), newlist.add(image), newlist)
  );
};

var S1_daily = ee.ImageCollection(
  ee.List(range.iterate(day_mosaics, ee.List([])))
);

print('Daily mosaics created:', S1_daily.size());


// ─── 4. COMPUTE DUAL-POL DESCRIPTORS PER IMAGE ───────────────────────────────

var descriptors = S1_daily.map(function(image) {

  // Convert dB → linear and apply spatial averaging (boxcar filter)
  var C11_mean = image.expression('10 ** (VV / 10)', {VV: image.select('VV')})
    .reduceNeighborhood({
      reducer: ee.Reducer.mean(),
      kernel:  ee.Kernel.square(window_size)
    });
  var C22_mean = image.expression('10 ** (VH / 10)', {VH: image.select('VH')})
    .reduceNeighborhood({
      reducer: ee.Reducer.mean(),
      kernel:  ee.Kernel.square(window_size)
    });

  // Derived quantities
  var span  = C11_mean.add(C22_mean);
  var ratio = C22_mean.divide(C11_mean);   // q = VH/VV

  // Vegetation mask: physically, VV should exceed VH
  var vmask = C11_mean.subtract(C22_mean).expression('b(0) > 0 ? 1 : 0');

  // ── mc: Co-pol purity parameter ──
  var mc = C11_mean.subtract(C22_mean).abs().divide(span);

  // ── Theta_c: Pseudo scattering type parameter (degrees) ──
  var Theta_c = C11_mean.subtract(C22_mean).abs()
    .multiply(span).multiply(mc)
    .divide(C11_mean.multiply(C22_mean).add(span.pow(2).multiply(mc.pow(2))))
    .atan()
    .multiply(180).divide(Math.PI);

  // ── Hc: Pseudo scattering entropy ──
  var p1  = C11_mean.divide(span);
  var p2  = C22_mean.divide(span);
  var log2 = ee.Number(2);
  var Hc  = p1.multiply(p1.log10()).divide(log2.log10()).multiply(-1)
    .add(p2.multiply(p2.log10()).divide(log2.log10()).multiply(-1));

  // ── DpRVIc: Dual-pol Radar Vegetation Index (corrected) ──
  // Formula: q(q+3) / (q+1)^2  where q = C22/C11
  var DpRVIc = ratio.multiply(ratio.add(3))
    .divide(ratio.add(1).multiply(ratio.add(1)));

  // ── Water mask: exclude pixels with VV < -17 dB ──
  var C11_dB   = C11_mean.log10().multiply(10);
  var watermask = C11_dB.expression('b(0) < -17 ? 0 : 1');

  // Apply masks
  mc      = mc.updateMask(vmask).updateMask(watermask);
  Hc      = Hc.updateMask(vmask).updateMask(watermask);
  Theta_c = Theta_c.updateMask(vmask).updateMask(watermask);
  DpRVIc  = DpRVIc.updateMask(vmask).updateMask(watermask);
  ratio   = ratio.updateMask(vmask).updateMask(watermask);

  // Stack all outputs
  var out = Hc.rename('Hc')
    .addBands(Theta_c.select('constant_mean').rename('Theta_c'))
    .addBands(mc.select('constant_mean').rename('mc'))
    .addBands(ratio.select('constant_mean').rename('ratio'))
    .addBands(C11_mean.select('constant_mean').rename('VV'))
    .addBands(C22_mean.select('constant_mean').rename('VH'))
    .addBands(DpRVIc.select('constant_mean').rename('DpRVIc'))
    .addBands(image.select('angle').rename('inc'));

  return out.set('system:time_start', image.get('system:time_start'));
});


// ─── 5. VISUALISE ────────────────────────────────────────────────────────────

var jet = ['#000080','#0000fa','#0057ff','#00c3ff','#3affbc',
           '#91ff66','#e8ff0f','#ffa400','#ff4000','#800000'];

Map.centerObject(extent, 10);
Map.addLayer(sample_pts, {color: 'red'}, 'TxSON Stations');
Map.addLayer(ee.Image(descriptors.select('DpRVIc').first()),
  {min: 0, max: 1, palette: jet}, 'DpRVIc');
Map.addLayer(ee.Image(descriptors.select('Hc').first()),
  {min: 0, max: 1, palette: jet}, 'Hc');
Map.addLayer(ee.Image(descriptors.select('mc').first()),
  {min: 0, max: 1, palette: jet}, 'mc');
Map.addLayer(ee.Image(descriptors.select('Theta_c').first()),
  {min: 0, max: 45, palette: jet}, 'Theta_c');


// ─── 6. EXTRACT TIME SERIES AT STATION LOCATIONS → CSV ───────────────────────

var bandcol  = ee.List(['Hc', 'Theta_c', 'mc', 'DpRVIc', 'ratio', 'VV', 'VH', 'inc']);
var bandsize = bandcol.size().getInfo();

for (var i = 0; i < bandsize; i++) {

  var band = ee.String(bandcol.get(i));

  var pts = sample_pts.map(function(feature) {
    return ee.Feature(feature.geometry(), {id: feature.id()});
  });

  var triplets = descriptors.map(function(image) {
    return image.select(band).reduceRegions({
      collection: pts,
      reducer:    ee.Reducer.first().setOutputs([band]),
      scale:      10
    }).map(function(feature) {
      var val = ee.List([feature.get(band), -9999]).reduce(ee.Reducer.firstNonNull());
      return feature.set({band: val, imageID: image.id()});
    });
  }).flatten();

  // Pivot: rows = stations, columns = acquisition dates
  var format = function(table, rowId, colId) {
    var rows = table.distinct(rowId);
    var joined = ee.Join.saveAll('matches').apply({
      primary:   rows,
      secondary: table,
      condition: ee.Filter.equals({leftField: rowId, rightField: rowId})
    });
    return joined.map(function(row) {
      var values = ee.List(row.get('matches')).map(function(f) {
        f = ee.Feature(f);
        return [f.get(colId), f.get(band)];
      });
      return row.select([rowId]).set(ee.Dictionary(values.flatten()));
    });
  };

  var results = format(triplets, 'id', 'imageID');

  Export.table.toDrive({
    collection:     results,
    description:    ee.String(band).cat('_time_series').getInfo(),
    folder:         'TxSON_S1_descriptors',
    fileNamePrefix: ee.String(band).cat('_time_series').getInfo(),
    fileFormat:     'CSV'
  });
}


// ─── 7. OPTIONAL: EXPORT FULL RASTER SCENES ──────────────────────────────────

// Uncomment to export all scenes as GeoTIFF (may take hours for 2015-2019)
/*
var ExportCol = function(col, folder, scale, type, nimg, maxPixels, region) {
  type      = type      || 'float';
  nimg      = nimg      || 500;
  scale     = scale     || 10;
  maxPixels = maxPixels || 1e12;
  var colList = col.toList(nimg);
  var n = colList.size().getInfo();
  for (var i = 0; i < n; i++) {
    var img = ee.Image(colList.get(i));
    var id  = img.id().getInfo();
    region  = region || img.geometry().bounds().getInfo()['coordinates'];
    Export.image.toDrive({
      image:         img.toFloat(),
      description:   id,
      folder:        folder,
      fileNamePrefix: id,
      region:        region,
      scale:         scale,
      maxPixels:     maxPixels
    });
  }
};
ExportCol(descriptors, 'TxSON_S1_rasters', 10, 'float', 500, 1e12, extent);
*/


/*=============================================================================
  REFERENCES

  Bhogapurapu, N., Dey, S., Homayouni, S., Bhattacharya, A., & Rao, Y.S.
    (2022). Field-scale soil moisture estimation using Sentinel-1 GRD SAR
    data. Advances in Space Research, 70(10), 3845-3858.
    https://doi.org/10.1016/j.asr.2022.03.019

  Caldwell, T., Bongiovanni, T., et al. (2019). Texas Soil Observation Network
    (TxSON). Texas Data Repository, V5.
    https://doi.org/10.18738/T8/JJ16CF

  Sentinel-1 GRD via Google Earth Engine:
    https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD
=============================================================================*/
