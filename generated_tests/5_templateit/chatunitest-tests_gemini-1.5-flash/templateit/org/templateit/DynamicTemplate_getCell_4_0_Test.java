package org.templateit;

import org.apache.poi.hssf.usermodel.HSSFCell;
import org.apache.poi.hssf.usermodel.HSSFRow;
import org.apache.poi.hssf.usermodel.HSSFSheet;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.List;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class DynamicTemplate_getCell_4_0_Test {

    @Mock
    private HSSFSheet sheet;

    @Mock
    private HSSFRow row;

    @Mock
    private HSSFCell cell;

    @Test
    void testGetCell_validCoordinates_returnsCell() throws NoSuchFieldException, IllegalAccessException, NoSuchMethodException, InvocationTargetException {
        // Arrange
        List<NamedStyle> styles = new ArrayList<>();
        DynamicTemplate template = new DynamicTemplate("test", sheet, 10, styles);
        // Setting width and height using reflection because they are private
        Field widthField = DynamicTemplate.class.getDeclaredField("width");
        widthField.setAccessible(true);
        widthField.setInt(template, 5);
        Field heightField = DynamicTemplate.class.getDeclaredField("height");
        heightField.setAccessible(true);
        heightField.setInt(template, 10);
        when(sheet.getRow(0)).thenReturn(row);
        when(row.getCell(0)).thenReturn(cell);
        // Act
        HSSFCell result = template.getCell(0, 0);
        // Assert
        assertEquals(cell, result);
    }

    @Test
    void testGetCell_invalidRow_returnsNull() throws NoSuchFieldException, IllegalAccessException, NoSuchMethodException, InvocationTargetException {
        // Arrange
        List<NamedStyle> styles = new ArrayList<>();
        DynamicTemplate template = new DynamicTemplate("test", sheet, 10, styles);
        Field widthField = DynamicTemplate.class.getDeclaredField("width");
        widthField.setAccessible(true);
        widthField.setInt(template, 5);
        Field heightField = DynamicTemplate.class.getDeclaredField("height");
        heightField.setAccessible(true);
        heightField.setInt(template, 10);
        // Act
        HSSFCell result = template.getCell(10, 0);
        // Assert
        assertNull(result);
    }

    @Test
    void testGetCell_invalidColumn_returnsNull() throws NoSuchFieldException, IllegalAccessException, NoSuchMethodException, InvocationTargetException {
        // Arrange
        List<NamedStyle> styles = new ArrayList<>();
        DynamicTemplate template = new DynamicTemplate("test", sheet, 10, styles);
        Field widthField = DynamicTemplate.class.getDeclaredField("width");
        widthField.setAccessible(true);
        widthField.setInt(template, 5);
        Field heightField = DynamicTemplate.class.getDeclaredField("height");
        heightField.setAccessible(true);
        heightField.setInt(template, 10);
        // Act
        HSSFCell result = template.getCell(0, 5);
        // Assert
        assertNull(result);
    }

    @Test
    void testGetCell_rowDoesNotExist_returnsNull() throws NoSuchFieldException, IllegalAccessException, NoSuchMethodException, InvocationTargetException {
        // Arrange
        List<NamedStyle> styles = new ArrayList<>();
        DynamicTemplate template = new DynamicTemplate("test", sheet, 10, styles);
        Field widthField = DynamicTemplate.class.getDeclaredField("width");
        widthField.setAccessible(true);
        widthField.setInt(template, 5);
        Field heightField = DynamicTemplate.class.getDeclaredField("height");
        heightField.setAccessible(true);
        heightField.setInt(template, 10);
        when(sheet.getRow(0)).thenReturn(null);
        // Act
        HSSFCell result = template.getCell(0, 0);
        // Assert
        assertNull(result);
    }
}
