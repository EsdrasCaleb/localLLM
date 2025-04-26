package org.templateit;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.List;
import org.apache.poi.hssf.usermodel.HSSFCell;
import org.apache.poi.hssf.usermodel.HSSFRow;
import org.apache.poi.hssf.usermodel.HSSFSheet;

@ExtendWith(MockitoExtension.class)
public class DynamicTemplate_height_0_0_Test {

    @Mock
    private HSSFSheet sheet;

    @Mock
    private List<NamedStyle> styles;

    private DynamicTemplate dynamicTemplate;

    @Test
    public void testHeight() {
        // Arrange
        dynamicTemplate = new DynamicTemplate("test", sheet, 10, styles);
        // Act
        int height = dynamicTemplate.height();
        // Assert
        assertEquals(10, height);
    }

    @Test
    public void testHeightWithNullStyles() {
        // Arrange
        dynamicTemplate = new DynamicTemplate("test", sheet, 10, null);
        // Act
        int height = dynamicTemplate.height();
        // Assert
        assertEquals(10, height);
    }

    @Test
    public void testHeightWithZeroHeight() {
        // Arrange
        dynamicTemplate = new DynamicTemplate("test", sheet, 0, styles);
        // Act
        int height = dynamicTemplate.height();
        // Assert
        assertEquals(0, height);
    }

    @Test
    public void testHeightWithNegativeHeight() {
        // Arrange
        dynamicTemplate = new DynamicTemplate("test", sheet, -10, styles);
        // Act
        int height = dynamicTemplate.height();
        // Assert
        assertEquals(0, height);
    }
}
