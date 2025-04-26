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

public class DynamicTemplate_width_1_0_Test {

    @Test
    public void testWidth() {
        // Arrange
        DynamicTemplate template = mock(DynamicTemplate.class);
        when(template.width()).thenReturn(5);
        // Act
        int width = template.width();
        // Assert
        assertEquals(5, width);
    }
}
