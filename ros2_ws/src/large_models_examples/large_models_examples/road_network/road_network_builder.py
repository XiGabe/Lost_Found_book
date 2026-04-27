#!/usr/bin/env python3
import cv2
import yaml
import math
import os
import sys
import copy

DISPLAY_SCALE = 2.0
DRAG_THRESHOLD = 5
DEFAULT_SAVE_NAME = "road_network.yaml"

class WaypointEditor:
    def __init__(self, map_yaml, save_name):
        self.save_name = '/home/ubuntu/ros2_ws/src/large_models_examples/large_models_examples/road_network/config/' + save_name
        self.edit_mode = False
        self.edit_target = None
        self.selected_id = None
        self.selected_from = None
        
        # 加载地图 / Load map
        with open(map_yaml, 'r') as f:
            map_data = yaml.safe_load(f)
        
        self.resolution = map_data['resolution']
        self.origin = map_data['origin']
        image_path = os.path.join(os.path.dirname(map_yaml),
                                  map_data['image'])
        
        self.map_raw = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.map_raw is None:
            print(f"错误: 无法加载地图图像: {image_path} (Error: Unable to load map image: {image_path})")
            sys.exit(1)
        self.map_raw = cv2.cvtColor(self.map_raw, cv2.COLOR_GRAY2BGR)
        
        # 尝试加载已有的路径点 / Try to load existing waypoints
        self.waypoints = []
        self.load_existing_waypoints()
        
        # 如果没有加载到，则创建起点 / If not loaded, create starting point
        if not self.waypoints:
            self.waypoints = [{
                'id': 0,
                'pose': [0.0, 0.0, 0.0],
                'to': []
            }]
        
        self.history = []
        self.redo_stack = []
        
        self.drag_start = None
        self.drag_current = None
        
        # 创建窗口 / Create window
        cv2.namedWindow("NetworkEditor", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("NetworkEditor",
                         int(self.map_raw.shape[1] * DISPLAY_SCALE),
                         int(self.map_raw.shape[0] * DISPLAY_SCALE))
        cv2.setMouseCallback("NetworkEditor", self.mouse_cb)
    
    def load_existing_waypoints(self):
        """加载已有的路径点文件 / Load existing waypoint file"""
        if os.path.exists(self.save_name):
            try:
                with open(self.save_name, 'r') as f:
                    data = yaml.safe_load(f)
                    if data and 'waypoints' in data:
                        self.waypoints = data['waypoints']
            except Exception as e:
                print(f"加载路径点错误: {e} (Error loading waypoints: {e})")
    
    # ---------- 工具函数 ---------- / ---------- Utility Functions ----------
    def push_history(self):
        self.history.append(copy.deepcopy(self.waypoints))
        self.redo_stack.clear()
    
    def undo(self):
        if self.history:
            self.redo_stack.append(copy.deepcopy(self.waypoints))
            self.waypoints = self.history.pop()
            print("撤销 (Undo)")
    
    def redo(self):
        if self.redo_stack:
            self.history.append(copy.deepcopy(self.waypoints))
            self.waypoints = self.redo_stack.pop()
            print("重做 (Redo)")
    
    def delete_selected(self):
        """删除选中的路径点，ID为0的点不允许删除 / Delete selected waypoint, ID 0 cannot be deleted"""
        if self.selected_id is not None:
            if self.selected_id == 0:
                print("错误: ID为0的基准点不能被删除 (Error: ID 0 reference point cannot be deleted)")
                return
            
            self.push_history()
            # 保存要删除的ID
            deleted_id = self.selected_id
            # 删除路径点
            del self.waypoints[deleted_id]
            
            # 重新分配ID
            for i, wp in enumerate(self.waypoints):
                wp['id'] = i
                # 更新连接关系
                wp['to'] = [t for t in wp['to'] if t != deleted_id]
                wp['to'] = [t if t < deleted_id else t-1 for t in wp['to']]
            
            print(f"删除路径点 {deleted_id} (Deleted waypoint {deleted_id})")
            self.selected_id = None
    
    # ---------- 坐标转换 ---------- / ---------- Coordinate Conversion ----------
    def pixel_to_world(self, px, py):
        h, _ = self.map_raw.shape[:2]
        x = px * self.resolution + self.origin[0]
        y = (h - py) * self.resolution + self.origin[1]
        return round(x, 2), round(y, 2)  # 保留2位小数 / Keep 2 decimal places
    
    def world_to_pixel(self, x, y):
        h, _ = self.map_raw.shape[:2]
        px = int((x - self.origin[0]) / self.resolution)
        py = int(h - (y - self.origin[1]) / self.resolution)
        return px, py
    
    def find_nearest(self, x, y, thresh=10):
        """查找最近的路径点 / Find nearest waypoint"""
        for wp in self.waypoints:
            px, py = self.world_to_pixel(wp['pose'][0], wp['pose'][1])
            if abs(px - x) < thresh and abs(py - y) < thresh:
                return wp['id']
        return None
    
    # ---------- 智能整理 ---------- / ---------- Smart Alignment ----------
    def smart_align(self):
        """智能整理：使相邻路径点对齐到垂直/水平方向 / Smart alignment: Align adjacent waypoints to vertical/horizontal directions"""
        if len(self.waypoints) < 2:
            print("没有路径点需要整理 (No waypoints to align)")
            return
        
        self.push_history()
        print("开始智能整理... (Starting smart alignment...)")
        
        # 遍历所有连接 / Iterate through all connections
        for wp in self.waypoints:
            px, py = wp['pose'][0], wp['pose'][1]
            
            for target_id in wp['to']:
                if target_id < len(self.waypoints):
                    target = self.waypoints[target_id]
                    tx, ty = target['pose'][0], target['pose'][1]
                    
                    # 计算连接的角度 / Calculate connection angle
                    dx = tx - px
                    dy = ty - py
                    dist = math.sqrt(dx*dx + dy*dy)
                    
                    if dist > 0:
                        # 判断是否接近垂直或水平 / Determine if close to vertical or horizontal
                        if abs(dx) > abs(dy):
                            # 接近水平，对齐y坐标 / Close to horizontal, align y coordinate
                            target['pose'][1] = round(py, 2)
                        else:
                            # 接近垂直，对齐x坐标 / Close to vertical, align x coordinate
                            target['pose'][0] = round(px, 2)
        
        print("智能整理完成 (Smart alignment completed)")
    
    # ---------- 鼠标事件 ---------- / ---------- Mouse Events ----------
    def mouse_cb(self, event, x, y, flags, param):
        rx, ry = int(x / DISPLAY_SCALE), int(y / DISPLAY_SCALE)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (rx, ry)
            self.drag_current = (rx, ry)
            
            # 检查是否点击了路径点 / Check if clicked on a waypoint
            nearest = self.find_nearest(rx, ry)
            if nearest is not None:
                self.selected_id = nearest
                if self.edit_mode:
                    # 检查是否为基准点（ID 0），基准点不能被编辑 / Check if it's reference point (ID 0), cannot be edited
                    if nearest == 0:
                        print("注意: ID为0的基准点不能被移动 (Note: ID 0 reference point cannot be moved)")
                        self.edit_target = None
                    else:
                        self.edit_target = nearest
                        print(f"选中路径点 {nearest} 进行编辑 (Selected waypoint {nearest} for editing)")
            else:
                self.selected_id = None
        
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start:
            self.drag_current = (rx, ry)
        
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start:
            dx = rx - self.drag_start[0]
            dy = ry - self.drag_start[1]
            dist = math.hypot(dx, dy)
            
            wx, wy = self.pixel_to_world(*self.drag_start)
            yaw = round(math.atan2(-dy, dx), 2) if dist > DRAG_THRESHOLD else None
            
            self.push_history()
            
            # 编辑模式 / Edit mode
            if self.edit_mode and self.edit_target is not None:
                # 再次确认不是基准点 / Double check it's not the reference point
                if self.edit_target == 0:
                    print("错误: ID为0的基准点不能被编辑 (Error: ID 0 reference point cannot be edited)")
                else:
                    wp = self.waypoints[self.edit_target]
                    wp['pose'][0] = wx
                    wp['pose'][1] = wy
                    if yaw is not None:
                        wp['pose'][2] = yaw
                    else:
                        wp['pose'][2] = 0.0
                    print(f"编辑路径点 {self.edit_target}: ({wx:.2f}, {wy:.2f}, {yaw or wp['pose'][2]:.2f}) (Edited waypoint {self.edit_target}: ({wx:.2f}, {wy:.2f}, {yaw or wp['pose'][2]:.2f}))")
            
            # 添加模式 / Add mode
            elif not self.edit_mode:
                new_id = len(self.waypoints)
                yaw_value = yaw if yaw is not None else 0.0
                self.waypoints.append({
                    'id': new_id,
                    'pose': [wx, wy, round(yaw_value, 2)],
                    'to': []
                })
                self.selected_id = new_id
                print(f"添加路径点 {new_id}: ({wx:.2f}, {wy:.2f}, {yaw_value:.2f}) (Added waypoint {new_id}: ({wx:.2f}, {wy:.2f}, {yaw_value:.2f}))")
            
            self.drag_start = None
            self.drag_current = None
            self.edit_target = None
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            idx = self.find_nearest(rx, ry)
            if idx is None:
                return
            
            if self.selected_from is None:
                self.selected_from = idx
                print(f"选择 {idx} 作为连接起点 (Selected {idx} as connection start)")
            else:
                if idx != self.selected_from:
                    self.push_history()
                    if idx not in self.waypoints[self.selected_from]['to']:
                        self.waypoints[self.selected_from]['to'].append(idx)
                        print(f"连接: {self.selected_from} -> {idx} (Connection: {self.selected_from} -> {idx})")
                    else:
                        # 如果连接已存在，删除它 / If connection already exists, remove it
                        self.waypoints[self.selected_from]['to'].remove(idx)
                        print(f"删除连接: {self.selected_from} -> {idx} (Removed connection: {self.selected_from} -> {idx})")
                self.selected_from = None
    
    # ---------- 绘制 ---------- / ---------- Drawing ----------
    def draw(self):
        img = self.map_raw.copy()
        
        # 先绘制连接线 / First draw connection lines
        for wp in self.waypoints:
            px, py = self.world_to_pixel(wp['pose'][0], wp['pose'][1])
            for t in wp['to']:
                if t < len(self.waypoints):
                    target = self.waypoints[t]
                    tx, ty = self.world_to_pixel(target['pose'][0], target['pose'][1])
                    cv2.line(img, (px, py), (tx, ty), (0, 255, 0), 1)
        
        # 绘制路径点 / Draw waypoints
        for wp in self.waypoints:
            px, py = self.world_to_pixel(wp['pose'][0], wp['pose'][1])
            
            # 颜色：起点绿色，选中黄色，其他红色 / Colors: start point green, selected yellow, others red
            if wp['id'] == 0:
                color = (0, 255, 0)  # 绿色 / Green - 基准点
                size = 5  # 基准点稍微大一点
            elif wp['id'] == self.selected_id:
                color = (0, 255, 255)  # 黄色 / Yellow
                size = 4
            else:
                color = (0, 0, 255)  # 红色 / Red
                size = 4
            
            # 绘制点 / Draw point
            cv2.circle(img, (px, py), size, color, -1)
            cv2.circle(img, (px, py), size+1, (255, 255, 255), 1)
            
            # 绘制方向箭头（小一点） / Draw direction arrow (smaller)
            try:
                yaw = wp['pose'][2]
            except:
                yaw = 0.0
            
            arrow_length = 12  # 减小箭头长度 / Reduce arrow length
            ax = int(px + arrow_length * math.cos(yaw))
            ay = int(py - arrow_length * math.sin(yaw))
            cv2.arrowedLine(img, (px, py), (ax, ay), (255, 0, 255), 1, tipLength=0.3)
            
            # 绘制ID（小一点，但保留在画面上） / Draw ID (smaller, but kept on screen)
            cv2.putText(img, str(wp['id']), (px+5, py-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
            
        
        # 绘制拖拽线 / Draw drag line
        if self.drag_start and self.drag_current:
            cv2.arrowedLine(img, self.drag_start, self.drag_current,
                           (0, 200, 255), 2, tipLength=0.2)
        
        return cv2.resize(img, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
    
    # ---------- 保存 ---------- / ---------- Save ----------
    def save(self):
        try:
            # 格式化数据：只包含waypoints / Format data: only include waypoints
            data_to_save = {'waypoints': []}
            
            for wp in self.waypoints:
                # 确保pose是两位小数 / Ensure pose has 2 decimal places
                pose = [
                    round(wp['pose'][0], 2),
                    round(wp['pose'][1], 2),
                    round(wp['pose'][2], 2)
                ]
                
                # 确保to列表中的ID是整数 / Ensure IDs in to list are integers
                to_list = [int(t) for t in wp['to']]
                
                data_to_save['waypoints'].append({
                    'id': int(wp['id']),
                    'pose': pose,
                    'to': to_list
                })
            
            # 保存到文件 / Save to file
            with open(self.save_name, 'w') as f:
                yaml.dump(data_to_save, f, sort_keys=False, default_flow_style=None)
            
            print(f"保存到 {self.save_name} (Saved to {self.save_name})")
            return True
            
        except Exception as e:
            print(f"保存错误: {e} (Save error: {e})")
            return False
    
    # ---------- 主循环 ---------- / ---------- Main Loop ----------
    def run(self):
        print("E: 切换编辑/添加模式 (Toggle Edit/Add Mode)")
        print("Z: 撤销 | Y: 重做 (Undo | Redo)")
        print("S: 保存 | L: 重新加载 (Save | Reload)")
        print("G: 智能整理 (Smart Align)")
        print("D: 删除选中的路径点 (Delete Selected Waypoint)")
        print("Q: 保存并退出 (Save and Quit)")
        print("ESC: 不保存退出 (Quit without Saving)")
        print("左键拖拽: 添加/编辑路径点 (Left drag: Add/Edit waypoint)")
        print("右键点击: 创建/删除连接 (Right click: Create/Remove connection)")

        while True:
            cv2.imshow("NetworkEditor", self.draw())
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('e'):
                self.edit_mode = not self.edit_mode
                mode_text = "编辑" if self.edit_mode else "添加"
                print(f"切换为{mode_text}模式 (Switched to {mode_text} mode)")
            
            elif key == ord('z'):
                self.undo()
            
            elif key == ord('y'):
                self.redo()
            
            elif key == ord('s'):
                self.save()
            
            elif key == ord('l'):
                self.load_existing_waypoints()
                print("重新加载路径点 (Reloaded waypoints)")
            
            elif key == ord('g'):
                self.smart_align()
            
            elif key == ord('d'):
                self.delete_selected()
            
            elif key == ord('q'):
                self.save()
                break
            
            elif key == 27:  # ESC
                print("退出程序 (Exiting program)")
                break
        
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # 命令行参数处理 / Command line argument processing
    if len(sys.argv) < 2:

        # 尝试默认路径 / Try default path
        default_map = "/home/ubuntu/ros2_ws/src/slam/maps/map_01.yaml"
        if os.path.exists(default_map):
            map_yaml = default_map
        else:
            print("请指定地图YAML文件 (Please specify map YAML file)")
            sys.exit(1)
    else:
        map_yaml = '/home/ubuntu/ros2_ws/src/slam/maps/' + str(sys.argv[1]) + '.yaml'

    save_name = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_SAVE_NAME
    
    # 创建编辑器实例 / Create editor instance
    editor = WaypointEditor(map_yaml, save_name)
    
    # 运行编辑器 / Run editor
    editor.run()
