package org.templateit;

import java.lang.reflect.Field;
import java.util.List;
import org.apache.poi.hssf.usermodel.HSSFSheet;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.apache.poi.hssf.usermodel.HSSFCell;
import org.apache.poi.hssf.usermodel.HSSFRow;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
class DynamicTemplate_height_0_1_Test {

    @Mock
    private HSSFSheet sheet;

    @Mock
    private List<NamedStyle> styles;

    private DynamicTemplate dynamicTemplate;

    @BeforeEach
    void setUp() {
        dynamicTemplate = new DynamicTemplate("Test", sheet, 5, styles);
    }

    @Test
    void testHeight() throws NoSuchFieldException, IllegalAccessException {
        // Set the private field height using reflection
        Field heightField = DynamicTemplate.class.getDeclaredField("height");
        heightField.setAccessible(true);
        heightField.set(dynamicTemplate, 10);
        // Invoke the height method and assert the result
        int height = dynamicTemplate.height();
        assertEquals(10, height);
    }

    @Test
    void testWidth() throws NoSuchFieldException, IllegalAccessException {
        // Set the private field width using reflection
        Field widthField = DynamicTemplate.class.getDeclaredField("width");
        widthField.setAccessible(true);
        widthField.set(dynamicTemplate, 10);
        // Invoke the width method and assert the result
        int width = dynamicTemplate.width();
        assertEquals(10, width);
    }

    @Test
    void testAbsoluteReference() throws NoSuchFieldException, IllegalAccessException {
        // Set the private fields using reflection
        Field widthField = DynamicTemplate.class.getDeclaredField("width");
        widthField.setAccessible(true);
        widthField.set(dynamicTemplate, 5);
        Field stylesField = DynamicTemplate.class.getDeclaredField("styles");
        stylesField.setAccessible(true);
        stylesField.set(dynamicTemplate, styles);
        NamedStyle style = mock(NamedStyle.class);
        when(styles.get(10)).thenReturn(style);
        when(style.getRow()).thenReturn(2);
        when(style.getColumn()).thenReturn(3);
        // Invoke the absoluteReference method and assert the result
        Reference reference = dynamicTemplate.absoluteReference(2, 0);
        assertEquals(2, reference.row());
        assertEquals(3, reference.column());
    }

    @Test
    void testGetRowHeight() throws NoSuchFieldException, IllegalAccessException {
        // Set the private fields using reflection
        Field widthField = DynamicTemplate.class.getDeclaredField("width");
        widthField.setAccessible(true);
        widthField.set(dynamicTemplate, 5);
        Field sheetField = DynamicTemplate.class.getDeclaredField("sheet");
        sheetField.setAccessible(true);
        sheetField.set(dynamicTemplate, sheet);
        HSSFRow row = mock(HSSFRow.class);
        when(sheet.getRow(2)).thenReturn(row);
        when(row.getHeight()).thenReturn((short) 100);
        // Invoke the getRowHeight method and assert the result
        int rowHeight = dynamicTemplate.getRowHeight(2);
        assertEquals(100, rowHeight);
    }

    @Test
    void testGetCell() throws NoSuchFieldException, IllegalAccessException {
        // Set the private fields using reflection
        Field widthField = DynamicTemplate.class.getDeclaredField("width");
        widthField.setAccessible(true);
        widthField.set(dynamicTemplate, 5);
        Field heightField = DynamicTemplate.class.getDeclaredField("height");
        heightField.setAccessible(true);
        heightField.set(dynamicTemplate, 5);
        Field sheetField = DynamicTemplate.class.getDeclaredField("sheet");
        sheetField.setAccessible(true);
        sheetField.set(dynamicTemplate, sheet);
        HSSFRow row = mock(HSSFRow.class);
        HSSFCell cell = mock(HSSFCell.class);
        when(sheet.getRow(2)).thenReturn(row);
        when(row.getCell(3)).thenReturn(cell);
        // Invoke the getCell method and assert the result
        HSSFCell resultCell = dynamicTemplate.getCell(2, 3);
        assertEquals(cell, resultCell);
    }
}
