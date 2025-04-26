package org.templateit;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.List;
import org.apache.poi.hssf.usermodel.HSSFCell;
import org.apache.poi.hssf.usermodel.HSSFRow;
import org.apache.poi.hssf.usermodel.HSSFSheet;

public class DynamicTemplate_width_1_1_Test {

    @Test
    void testWidth() {
        // Arrange
        DynamicTemplate dynamicTemplate = mock(DynamicTemplate.class);
        int expectedWidth = 10;
        when(dynamicTemplate.width()).thenReturn(expectedWidth);
        // Act
        int actualWidth = dynamicTemplate.width();
        // Assert
        assertEquals(expectedWidth, actualWidth);
    }
}
