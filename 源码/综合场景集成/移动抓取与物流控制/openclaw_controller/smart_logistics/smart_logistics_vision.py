import time

import cv2
import numpy as np
import sdk.common as common


class SmartLogisticsVisionMixin:
    def display_loop(self):
        while self.running:
            with self.image_lock:
                if self.latest_image is None:
                    time.sleep(0.01)
                    continue
                img = self.latest_image.copy()
            result_image = self.draw_result_image(img)
            cv2.imshow('smart_logistics', result_image)
            cv2.waitKey(1)

    def draw_result_image(self, image):
        result_image = image.copy()
        result = self.display_result
        if result is None:
            return result_image

        x_min, y_min, x_max, y_max = result['roi']
        cv2.rectangle(result_image, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
        if 'items' in result:
            for item in result['items']:
                for box, center in zip(item['boxes'], item['centers']):
                    cv2.drawContours(result_image, [box], -1, common.range_rgb[item['color']], 2)
                    cv2.circle(result_image, center, 5, common.range_rgb[item['color']], -1)
        elif result['box'] is not None:
            cv2.drawContours(result_image, [result['box']], -1, common.range_rgb[result['color']], 2)
            cv2.circle(result_image, result['center'], 5, common.range_rgb[result['color']], -1)
        text = result.get('text', result['color'])
        cv2.putText(result_image, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        return result_image

    def publish_result_image(self, image):
        result_image = self.draw_result_image(image)
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(result_image, 'bgr8'))

    def clear_display_later(self, result, delay=3.0):
        time.sleep(delay)
        if self.display_result is result:
            self.display_result = None

    def get_roi(self, image, roi=None):
        roi = self.roi if roi is None else roi
        h, w = image.shape[:2]
        x_min = int(w * roi[0])
        y_min = int(h * roi[1])
        x_max = int(w * roi[2])
        y_max = int(h * roi[3])
        x_min = max(0, min(x_min, w - 1))
        y_min = max(0, min(y_min, h - 1))
        x_max = max(x_min + 1, min(x_max, w))
        y_max = max(y_min + 1, min(y_max, h))
        return x_min, y_min, x_max, y_max

    def get_task_roi(self, task_name):
        if task_name in ['pick', 'pick_stop_pixel_coordinate']:
            scale = self.pick_roi_scale
            x_min = (1.0 - scale) / 2.0
            y_min = (1.0 - scale) / 2.0
            return [x_min, y_min, x_min + scale, y_min + scale]
        scale = self.place_roi_scale
        x_min = (1.0 - scale) / 2.0
        y_min = (1.0 - scale) / 2.0
        return [x_min, y_min, x_min + scale, 0.5]

    def get_color_contours(self, image, color, roi=None):
        x_min, y_min, x_max, y_max = self.get_roi(image, roi)
        roi = image[y_min:y_max, x_min:x_max]
        img_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        img_blur = cv2.GaussianBlur(img_lab, (3, 3), 3)
        color_range = self.lab_data['lab'][self.camera_type][color]
        mask = cv2.inRange(img_blur, tuple(color_range['min']), tuple(color_range['max']))
        eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)[-2]
        return contours, (x_min, y_min, x_max, y_max)

    def count_target_color(self, image, color, roi=None):
        contours, roi = self.get_color_contours(image, color, roi)
        x_min, y_min, x_max, y_max = roi
        count = 0
        boxes = []
        centers = []
        for contour in contours:
            if cv2.contourArea(contour) >= self.min_area:
                count += 1
                rect = cv2.minAreaRect(contour)
                box = np.intp(cv2.boxPoints(rect))
                box = box + np.array([x_min, y_min], dtype=box.dtype)
                boxes.append(box)
                centers.append((int(rect[0][0] + x_min), int(rect[0][1] + y_min)))
        return count, {
            'color': color, 'count': count,
            'roi': (x_min, y_min, x_max, y_max),
            'boxes': boxes, 'centers': centers,
            'box': None, 'center': (-1, -1), 'angle': -1,
        }

    def count_all_colors(self, image, roi=None):
        items = []
        counts = {}
        result_roi = None
        for color in self.color_names:
            count, result = self.count_target_color(image, color, roi)
            counts[color] = count
            result_roi = result['roi']
            items.append({
                'color': color,
                'count': count,
                'boxes': result['boxes'],
                'centers': result['centers'],
            })
        text = ','.join(['%s:%d' % (color, counts[color]) for color in self.color_names])
        return counts, {
            'color': 'red',
            'text': text,
            'roi': result_roi,
            'items': items,
            'box': None, 'center': (-1, -1), 'angle': -1,
        }

    def detect_target_color(self, image, color, roi=None):
        contours, roi = self.get_color_contours(image, color, roi)
        x_min, y_min, x_max, y_max = roi
        max_contour, area = common.get_area_max_contour(contours, self.min_area)
        if max_contour is None:
            return self.empty_result((x_min, y_min, x_max, y_max), area)

        rect = cv2.minAreaRect(max_contour)
        box = np.intp(cv2.boxPoints(rect))
        box = box + np.array([x_min, y_min], dtype=box.dtype)
        center = (int(rect[0][0] + x_min), int(rect[0][1] + y_min))
        return {
            'color': color, 'area': area,
            'roi': (x_min, y_min, x_max, y_max),
            'box': box, 'center': center, 'angle': rect[2],
        }

    def empty_result(self, roi, area=0):
        return {
            'color': 'none', 'area': area, 'roi': roi,
            'box': None, 'center': (-1, -1), 'angle': -1,
        }

    def image_callback(self, ros_image):
        image = self.bridge.imgmsg_to_cv2(ros_image, 'bgr8')
        image = np.array(image, dtype=np.uint8)
        with self.image_lock:
            self.latest_image = image
        self.publish_result_image(image)
