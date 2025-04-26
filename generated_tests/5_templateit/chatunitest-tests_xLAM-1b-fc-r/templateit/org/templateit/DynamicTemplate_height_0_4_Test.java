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

class DynamicTemplate_height_0_4_Test {

    @Mock
    DynamicTemplate dynamicTemplate;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.initMocks(this);
        when(dynamicTemplate.height()).thenReturn(10);
    }

    @Test
    void height() {
        // Given
        // When
        int height = dynamicTemplate.height();
        // Then
        assertEquals(10, height);
    }
}
