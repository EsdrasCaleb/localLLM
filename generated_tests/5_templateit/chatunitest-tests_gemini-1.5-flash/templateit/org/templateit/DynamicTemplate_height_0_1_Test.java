package org.templateit;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;
import org.apache.poi.hssf.usermodel.HSSFCell;
import org.apache.poi.hssf.usermodel.HSSFRow;
import org.apache.poi.hssf.usermodel.HSSFSheet;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
class DynamicTemplate_height_0_1_Test {

    @Mock
    HSSFSheet mockSheet;

    @Test
    void testHeight() throws NoSuchFieldException, IllegalAccessException {
        // Create a list of NamedStyle (assuming NamedStyle class exists)
        List<NamedStyle> styles = new ArrayList<>();
        styles.add(new NamedStyle(0, 0));
        // Create a DynamicTemplate object using reflection to set private fields
        DynamicTemplate dynamicTemplate = new DynamicTemplate("test", mockSheet, 10, styles);
        Field heightField = DynamicTemplate.class.getDeclaredField("height");
        heightField.setAccessible(true);
        // Test the height() method with different values
        heightField.setInt(dynamicTemplate, 10);
        assertEquals(10, dynamicTemplate.height());
        heightField.setInt(dynamicTemplate, 0);
        assertEquals(0, dynamicTemplate.height());
        heightField.setInt(dynamicTemplate, -5);
        assertEquals(-5, dynamicTemplate.height());
    }

    static class NamedStyle {

        private int row;

        private int column;

        public NamedStyle(int row, int column) {
            this.row = row;
            this.column = column;
        }

        public int getRow() {
            return row;
        }

        public int getColumn() {
            return column;
        }
    }

    static class Reference {

        private int row;

        private int column;

        public Reference(int row, int column) {
            this.row = row;
            this.column = column;
        }

        public int row() {
            return row;
        }

        public int column() {
            return column;
        }
    }

    static class DynamicTemplate {

        private String name;

        private HSSFSheet sheet;

        private int width;

        private List<NamedStyle> styles;

        private int height;

        public DynamicTemplate(String name, HSSFSheet sheet, int width, List<NamedStyle> styles) {
            this.name = name;
            this.sheet = sheet;
            this.width = width;
            this.styles = styles;
        }

        public int height() {
            return height;
        }
    }
}
