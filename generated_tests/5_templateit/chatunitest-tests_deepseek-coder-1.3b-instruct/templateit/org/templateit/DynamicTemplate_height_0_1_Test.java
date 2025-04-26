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

public class DynamicTemplate_height_0_1_Test {

    @Test
    public void testHeight() {
        // Arrange
        DynamicTemplate dynamicTemplate = Mockito.mock(DynamicTemplate.class);
        Mockito.when(dynamicTemplate.height()).thenReturn(10);
        // Act
        int result = dynamicTemplate.height();
        // Assert
        assertEquals(10, result);
    }
}
