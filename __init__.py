from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np
import cv2
from shapely.wkt import loads as wkt_loads
from shapely.geometry import Polygon

def _get_polygon_list(wkt_list_pandas, imageId):
    df_image = wkt_list_pandas[wkt_list_pandas.ImageId == imageId]
    poly_def = df_image.PolygonWKT_Pix
    polygonList = None
    polygonList = [wkt_loads(x) for x in poly_def]
    return polygonList, poly_def


def _get_and_convert_contours(polygonList, raster_img_size, poly_def):
    perim_list = []
    interior_list = []
    if len(poly_def) < 2:
        return None
    for k in range(len(poly_def)):
        poly = polygonList[k]
        perim = np.array(list(poly.exterior.coords))
        #perim_c = np.array(perim[:,:-1]).astype(int)
        perim_c = np.array(perim[:,:]).astype(int)
        perim_list.append(perim_c)
        for pi in poly.interiors:
            interior = np.array(list(pi.coords))
#            interior_c = _convert_coordinates_to_raster(interior, raster_img_size)
            interior_list.append(np.int32(interior[:,:-1]))
    return perim_list,interior_list


def _plot_mask_from_contours(raster_img_size, ext_pts, int_pts, class_value = 1):
    img_mask = np.zeros(raster_img_size,np.uint8)

    ext_pointsnp = np.asarray(ext_pts)
    cv2.fillPoly(img_mask, np.asarray(ext_pts),1)
    cv2.fillPoly(img_mask, np.asarray(int_pts),0)
    return img_mask

def generate_mask_for_image_and_class(raster_size, imageId, wkt_list_pandas):
    polygon_list, poly_def = _get_polygon_list(wkt_list_pandas,imageId)
    if len(polygon_list) < 2:
        mask = np.zeros(raster_size,np.uint8)
        return mask
    else:
        ext, inte = _get_and_convert_contours(polygon_list,raster_size, poly_def)
        mask = _plot_mask_from_contours(raster_size,ext, inte,1)
        return mask