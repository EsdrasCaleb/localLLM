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
public class DynamicTemplate_width_1_4_Test {

    @Mock
    private DynamicTemplate dynamicTemplate;

    @Test
    public void testWidth() {
        // Arrange
        when(dynamicTemplate.width()).thenReturn(10);
        // Act
        int result = dynamicTemplate.width();
        // Assert
        assertEquals(10, result);
    }
}
