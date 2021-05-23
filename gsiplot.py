import numpy as np
import pandas as pd
import math
from PIL import Image
import requests
import io
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import patheffects as path_effects


def is_num(s):
    try:
        float(s)
    except ValueError:
        return False
    else:
        return True

def get_num_order(x):
    head_flg = 0
    period_flg = 0
    order = 0
    x_str = f'{x:f}'

    if x_str[0] == '-':
        x_str = x_str[1:]
    for c in x_str:
        assert is_num(c) or c == '.', f'Unexpected character: {c}'
        if not period_flg:
            if head_flg:
                if c == '.':
                    return order
                else:
                    order += 1
            else:
                if c == '.':
                    period_flg = 1
                elif not c == '0':
                    head_flg = 1
        else:
            assert not c == '.', 'Duplicate periods.'
            order -= 1
            if not c == '0':
                return order
    if head_flg:
        return order
    else:
        return 0

def location2tile(z, lon, lat):
    x = int((lon / 180 + 1) * 2**z / 2)
    y = int(((-math.log(math.tan((45 + lat / 2) * math.pi / 180)) + math.pi) * 2**z / (2 * math.pi)))
    return z,x,y

def tile2location(z, x, y):
	lon = (x / 2.0**z) * 360 - 180 # 経度（東経）
	mapy = (y / 2.0**z) * 2 * math.pi - math.pi
	lat = 2 * math.atan(math.e ** (- mapy)) * 180 / math.pi - 90 # 緯度（北緯）
	return lat,lon

def lon_formatter(x, pos):
    if x == 0:
        return '0°'
    elif 0 < x and x < 180:
        return f'{x}°E'
    elif x == 180:
        return '180°'
    else:
        return f'{-1 * x}°W'

def lat_formatter(x, pos):
    if x == 0:
        return 'EQ'
    elif x > 0:
        return f'{x}°N'
    else:
        return f'{-1 * x}°S'

lat_bases = [180 / 2**n for n in np.arange(0, 19)]
lon_bases = [360 / 2**n for n in np.arange(0, 19)]

###############################################
#
#   DEM
#
###############################################
class dem:
    def __init__(self,map_NW,map_SE,zoom_mode=0):
        self.map_NW = map_NW
        self.map_SE = map_SE
        dist_lon = map_SE[1] - map_NW[1]
        dist_lat = map_NW[0] - map_SE[0]
        assert dist_lon > 0 and dist_lat > 0, "The coordinates are wrong."
        self.zoom = 18
        if zoom_mode == 0:
            zoom_list = np.arange(18, -1, -1)
            for n in zoom_list:
                if dist_lon <= lon_bases[n] or dist_lat <= lat_bases[n]:
                    self.zoom = n + 1
                    assert 15 > self.zoom and self.zoom > 0, "The coordinates are too near."
                    break
        else:
            self.zoom = zoom_mode
        tile_NW = location2tile(self.zoom, map_NW[1], map_NW[0])
        tile_SE = location2tile(self.zoom, map_SE[1], map_SE[0])
        self.dem_NW = tile2location(*tile_NW)
        self.dem_SE = tile2location(tile_SE[0], tile_SE[1]+1, tile_SE[2]+1)
        print(f'zoom: {self.zoom}')
        self.all_dem = self.fetch_all_tiles(tile_NW, tile_SE)
        print(f'Download finished.')
        self.Z_max = np.amax(self.all_dem)
        self.Z_min = np.amin(self.all_dem[~(self.all_dem == -500)])
        self.Z = np.where(self.all_dem == -500, self.Z_min-1, self.all_dem)
        self.land_area = np.where(self.all_dem == -500, 0, 1)
        self.X, self.Y = np.meshgrid(np.linspace(self.dem_NW[1], self.dem_SE[1], self.Z.shape[1]), np.linspace(self.dem_NW[0], self.dem_SE[0], self.Z.shape[0]))
        print(f'max elev: {self.Z_max}, min elev: {self.Z_min}')

        self.dpi = 300
        self.font_family = 'Noto Sans CJK JP'
        self.lat_ticks = 5
        self.lon_ticks = 4
        self.lat_label = "緯度"
        self.lon_label = "経度"
        self.cbar_label = '標高 [m]'
        self.shade = True
        self.coastline = False
        self.contour_label = False
        self.contour = 100
        self.sub_contour = 20
        self.route_lat = []
        self.route_lon = []
        self.scatters = []

    def fetch_tile(self, z, x, y):
        url = f"https://cyberjapandata.gsi.go.jp/xyz/dem/{z}/{x}/{y}.txt"
        df =  pd.read_csv(url, header=None)
        pic = df.applymap(lambda s: float(s) if is_num(s) else -500)
        return pic.values

    def fetch_all_tiles(self, north_west, south_east):
        assert north_west[0] == south_east[0], "zoom index is different between tiles!"
        x_range = range(north_west[1], south_east[1]+1)
        y_range = range(north_west[2], south_east[2]+1)
        return  np.concatenate([np.concatenate([self.fetch_tile(north_west[0], x, y) for y in y_range], axis=0) for x in x_range], axis=1)

    def hillshade(self, Z, azimuth_deg=315, altitude_deg=45, z_factor=1):
        Zx, Zy = np.gradient(Z)
        Slope_rad = np.arctan(z_factor * np.sqrt(Zx*Zx + Zy*Zy))
        Aspect_rad = np.arctan2(-Zx, Zy)
        Azimuth_rad = np.radians(360.0 - azimuth_deg + 90)
        Zenith_rad = np.radians(90 - altitude_deg)
        shaded = np.cos(Zenith_rad) * np.cos(Slope_rad) + np.sin(Zenith_rad) * np.sin(Slope_rad) * np.cos(Azimuth_rad - Aspect_rad)
        return 255 * shaded

    def makefig(self, filename='./result.png', figtitle=''):
        matplotlib.rc('font', family=self.font_family)
        fig = plt.figure(dpi=self.dpi, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(111, aspect="equal")

        colors_undersea = plt.cm.terrain(np.linspace(0, 0.17, 256))
        colors_land = plt.cm.terrain(np.linspace(0.25, 1, 256))
        all_colors = np.vstack((colors_undersea, colors_land))
        terrain_map = colors.LinearSegmentedColormap.from_list('terrain_map', all_colors)
        divnorm = colors.TwoSlopeNorm(vmin=self.Z_min-2, vcenter=self.Z_min, vmax=self.Z_max)

        pcolor = ax.pcolormesh(self.X, self.Y, self.Z, shading='gouraud', norm=divnorm, cmap=terrain_map, zorder=1)

        if self.shade:
            hillshade = self.hillshade(self.Z, 180, 40) * self.land_area
            shade_colors = [(0.3, 0.3, 0.3, n*0.1) for n in np.linspace(0, 1, 256)]
            shade_map = colors.LinearSegmentedColormap.from_list('shade_map', shade_colors)
            shaded = ax.pcolormesh(self.X, self.Y, hillshade, vmin=0, vmax=1, shading='gouraud', cmap=shade_map, zorder=2)

        plt.xlim(self.map_SE[1], self.map_NW[1])
        plt.ylim(self.map_SE[0], self.map_NW[0])
        ax.set_xlabel(self.lon_label, fontsize=12)
        ax.set_ylabel(self.lat_label, fontsize=12)
        ax.invert_xaxis()

        # 軸のフォーマット
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        lat_dist = self.map_NW[0] - self.map_SE[0] 
        lon_dist = self.map_SE[1] - self.map_NW[1]
        lat_order = get_num_order(lat_dist) -1
        lon_order = get_num_order(lon_dist) -1
        lat_min = math.ceil(self.map_SE[0] / 10 ** lat_order) * 10 ** lat_order
        lat_max = math.floor(self.map_NW[0] / 10 ** lat_order) * 10 ** lat_order
        lon_min = math.ceil(self.map_NW[1] / 10 ** lon_order) * 10 ** lon_order
        lon_max = math.floor(self.map_SE[1] / 10 ** lon_order) * 10 ** lon_order
        lat_ticks = np.linspace(lat_min, lat_max, self.lat_ticks)
        lon_ticks = np.linspace(lon_min, lon_max, self.lon_ticks)
        if lon_order < 0:
            lon_ticks = [round(x, -1*lon_order) for x in lon_ticks]
        if lat_order < 0:
            lat_ticks = [round(x, -1*lat_order) for x in lat_ticks]
        ax.set_xticks(lon_ticks)
        ax.set_yticks(lat_ticks)

        # 出典表記
        text_copyright = ax.text(0.99, 0.02, '出典：国土地理院',
                c='white',alpha=1, fontsize=6, zorder=10,
                ha='right', transform=ax.transAxes)
        text_copyright.set_path_effects([
                path_effects.Stroke(linewidth=0.7, foreground='black'),
                path_effects.Normal()])

        if self.sub_contour > 0:
            sub_cont_min = math.ceil(self.Z_min / self.sub_contour) * self.sub_contour
            sub_cont_max = math.ceil(self.Z_max / self.sub_contour) * self.sub_contour
            sub_elevation = np.arange(sub_cont_min, sub_cont_max, self.sub_contour)
            sub_cont = ax.contour(self.X, self.Y, self.Z, levels=sub_elevation, colors=['brown'], linewidths=0.1, zorder=3)
        if self.contour > 0:
            cont_min = math.ceil(self.Z_min / self.contour) * self.contour
            cont_max = math.ceil(self.Z_max / self.contour) * self.contour
            elevation = np.arange(cont_min, cont_max, self.contour)
            cont = ax.contour(self.X, self.Y, self.Z, levels=elevation, colors=['brown'], linewidths=0.3, zorder=4)
            if self.contour_label:
                cont.clabel(fmt='%1d', fontsize=5, colors=['#402700'])
        if self.coastline:
            coast_lev = [0.5]
            coastline = ax.contour(self.X, self.Y, self.land_area, levels=coast_lev, colors=['black'], linewidths=0.5, zorder=5)

        if self.route_lat != [] and self.route_lon != []:
            # 軌跡
            s_xytext = [np.nan, np.nan]
            first_flg = True
            for s_lat, s_lon in zip (self.scatter_lat, self.scatter_lon):
                if not first_flg:
                    ax.annotate('', xy=[s_lon, s_lat], xytext=s_xytext,
                            arrowprops=dict(shrink=0, width=0.5, headwidth=6, 
                            headlength=7, connectionstyle='arc3',
                            facecolor='royalblue', edgecolor='cyan'), zorder=6)
                else:
                    first_flg = False
                s_xytext = [s_lon, s_lat]
            srt_end_lat = [self.scatter_lat[0], self.scatter_lat[-1]]
            srt_end_lon = [self.scatter_lon[0], self.scatter_lon[-1]]
            ax.scatter(srt_end_lon, srt_end_lat, s=150, marker='*', c='blue', edgecolors="yellow", linewidths=0.8, zorder=7)

        if len(self.scatters) > 0:
            # scatter
            for scatter in self.scatters:
                scat_marker = scatter['marker'] if 'marker' in scatter.keys() else 'o'
                scat_color = scatter['c'] if 'c' in scatter.keys() else 'black'
                scat_size = scatter['s'] if 's' in scatter.keys() else 50
                ax.scatter(scatter['lon'], scatter['lat'], marker=scat_marker, c=scat_color, s=scat_size, edgecolor='white', linewidths=0.6, zorder=8)

        cbar = fig.colorbar(pcolor, shrink=0.8, aspect=25, orientation='vertical')
        cbar.set_label(self.cbar_label, fontsize=15)
        cbar.ax.tick_params(labelsize=10)

        if figtitle != '':
            plt.title(figtitle)

        plt.savefig(filename, bbox_inches='tight', pad_inches=0.05)
        plt.clf()
        plt.close()
        print(f'### END! a figure saved as "{filename}" ###')

###############################################
#
#   Aerial photos
#
###############################################
class photo:
    def __init__(self,map_NW, map_SE, zoom_mode=0):
        self.map_NW = map_NW
        self.map_SE = map_SE
        dist_lon = map_SE[1] - map_NW[1]
        dist_lat = map_NW[0] - map_SE[0]
        assert dist_lon > 0 and dist_lat > 0, "The coordinates are wrong."
        # zoom の設定
        self.zoom = 18
        if zoom_mode == 0:
            zoom_list = np.arange(18, -1, -1)
            for n in zoom_list:
                if dist_lon <= lon_bases[n] or dist_lat <= lat_bases[n]:
                    self.zoom = n + 1
                    assert self.zoom > 1, f"Zoom level {self.zoom} is not supported."
                    break
        else:
            self.zoom = zoom_mode
        print(f'zoom: {self.zoom}')
        # タイル座標の取得
        tile_NW = location2tile(self.zoom, map_NW[1], map_NW[0])
        tile_SE = location2tile(self.zoom, map_SE[1], map_SE[0])
        self.photo_NW = tile2location(*tile_NW)
        self.photo_SE = tile2location(tile_SE[0], tile_SE[1]+1, tile_SE[2]+1)
        # DEMデータのダウンロード
        self.all_photo = self.get_all_aerialphotos(tile_NW, tile_SE)
        print(f'Download finished.')

        self.dpi = 300
        self.font_family = 'Noto Sans CJK JP'
        self.lat_ticks = 5
        self.lon_ticks = 4
        self.lat_label = "緯度"
        self.lon_label = "経度"
        self.scatter_lat = []
        self.scatter_lon = []
        self.scatter_val = []
        self.scatter_cmap = 'jet'
        self.scatter_vmin = 0
        self.scatter_vmax = 50
        self.cbar_label = ''

    def get_aerialphoto(self, z, x, y):
        url = f"https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg"
        req = requests.get(url)
        im = Image.open( io.BytesIO(req.content) )
        im_list = np.asarray(im)
        return im_list

    def get_all_aerialphotos(self, north_west, south_east):
        assert north_west[0] == south_east[0], "zoom index is different between tiles!"
        x_range = range(north_west[1], south_east[1]+1)
        y_range = range(north_west[2], south_east[2]+1)
        return  np.concatenate([np.concatenate([self.get_aerialphoto(north_west[0], x, y) for y in y_range], axis=0) for x in x_range], axis=1)

    def makefig(self, filename='./result.png', figtitle=''):
        matplotlib.rc('font', family=self.font_family)
        fig = plt.figure(dpi=self.dpi, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(111, aspect="equal")

        # 表示範囲
        ax.set_xlim(self.map_NW[1], self.map_SE[1])
        ax.set_ylim(self.map_SE[0], self.map_NW[0])
        ax.set_xlabel(self.lon_label, fontsize=12)
        ax.set_ylabel(self.lat_label, fontsize=12)

        # 軸のフォーマット
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        lat_dist = self.map_NW[0] - self.map_SE[0] 
        lon_dist = self.map_SE[1] - self.map_NW[1]
        lat_order = get_num_order(lat_dist) -1
        lon_order = get_num_order(lon_dist) -1
        lat_min = math.ceil(self.map_SE[0] / 10 ** lat_order) * 10 ** lat_order
        lat_max = math.floor(self.map_NW[0] / 10 ** lat_order) * 10 ** lat_order
        lon_min = math.ceil(self.map_NW[1] / 10 ** lon_order) * 10 ** lon_order
        lon_max = math.floor(self.map_SE[1] / 10 ** lon_order) * 10 ** lon_order
        lat_ticks = np.linspace(lat_min, lat_max, self.lat_ticks)
        lon_ticks = np.linspace(lon_min, lon_max, self.lon_ticks)
        if lon_order < 0:
            lon_ticks = [round(x, -1*lon_order) for x in lon_ticks]
        if lat_order < 0:
            lat_ticks = [round(x, -1*lat_order) for x in lat_ticks]
        ax.set_xticks(lon_ticks)
        ax.set_yticks(lat_ticks)

        # 画像を表示
        photo_extent = (self.photo_NW[1], self.photo_SE[1], self.photo_SE[0], self.photo_NW[0])
        ax.imshow(self.all_photo, origin='upper', extent=photo_extent)
        # 出典表記
        text_copyright = ax.text(0.99, 0.02, '出典：国土地理院',
                c='white',alpha=0.8, fontsize=6, zorder=10,
                ha='right', transform=ax.transAxes)
        text_copyright.set_path_effects([
                path_effects.Stroke(linewidth=1, foreground='black'),
                path_effects.Normal()])

        if 9 <= self.zoom and self.zoom <= 13:
            text_bottom = 'データソース：Landsat8画像（GSI,TSIC,GEO Grid/AIST）, \nLandsat8画像（courtesy of the U.S. Geological Survey）, 海底地形（GEBCO）'
            size_bottom = 0.16
            fig.subplots_adjust(left=0, bottom=size_bottom)
            fig.text(0.2, 0.02, text_bottom, c='gray', fontsize=5, ha='left')
        elif 2 <= self.zoom and self.zoom <= 8:
            text_bottom = 'Images on 世界衛星モザイク画像 obtained from site https://lpdaac.usgs.gov/data_access \nmaintained by the NASA Land Processes Distributed Active Archive Center (LP DAAC), \nUSGS/Earth Resources Observation and Science (EROS) Center, Sioux Falls, \nSouth Dakota, (Year). Source of image data product.'
            size_bottom = 0.2
            fig.subplots_adjust(left=0, bottom=size_bottom)
            fig.text(0.2, 0.02, text_bottom, c='gray', fontsize=5, ha='left')

        if self.scatter_lat != [] and self.scatter_lon != []:
            if self.scatter_val == []:
                # 軌跡
                s_xytext = [np.nan, np.nan]
                first_flg = True
                for s_lat, s_lon in zip (self.scatter_lat, self.scatter_lon):
                    if not first_flg:
                        ax.annotate('', xy=[s_lon, s_lat], xytext=s_xytext,
                                arrowprops=dict(shrink=0, width=0.5, headwidth=6, 
                                headlength=7, connectionstyle='arc3',
                                facecolor='royalblue', edgecolor='cyan'), zorder=2)
                    else:
                        first_flg = False
                    s_xytext = [s_lon, s_lat]
                srt_end_lat = [self.scatter_lat[0], self.scatter_lat[-1]]
                srt_end_lon = [self.scatter_lon[0], self.scatter_lon[-1]]
                ax.scatter(srt_end_lon, srt_end_lat, s=150, marker='*', c='blue', edgecolors="yellow", linewidths=0.8, zorder=3)
            else:
                # データ
                points = ax.scatter(self.scatter_lon, self.scatter_lat,
                        c=self.scatter_val, cmap=self.scatter_cmap, 
                        vmin=self.scatter_vmin, vmax=self.scatter_vmax,
                        s=40, marker='o', edgecolor='white', linewidths=0.6)
                if self.cbar_label != '':
                    cbar = fig.colorbar(points, shrink=0.8, aspect=25, orientation='vertical')
                    cbar.set_label(self.cbar_label, fontsize=15)
                    cbar.ax.tick_params(labelsize=10)
        
        if figtitle != '':
            plt.title(figtitle)
                    
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.05)
        plt.clf()
        plt.close()
        print(f'### END! a figure saved as "{filename}" ###')