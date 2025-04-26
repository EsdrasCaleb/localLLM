package org.templateit;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.junit.MockitoJUnitRunner;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
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

@RunWith(MockitoJUnitRunner.class)
public class DynamicTemplate_height_0_1_Test {

    @InjectMocks
    private DynamicTemplate dynamicTemplate;

    @Mock
    private HSSFSheet sheet;

    @Mock
    private NamedStyle styles;

    @Test
    public void testHeight() {
        // Arrange
        when(dynamicTemplate.getRowHeight(0)).thenReturn(10);
        when(dynamicTemplate.getRowHeight(1)).thenReturn(20);
        // Act
        int height = dynamicTemplate.height();
        // Assert
        assertEquals(20, height, 0);
    }
}
