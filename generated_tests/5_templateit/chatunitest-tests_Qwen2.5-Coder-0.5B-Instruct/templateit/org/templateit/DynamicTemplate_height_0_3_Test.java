package org.templateit;

import org.templateit.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.List;
import org.apache.poi.hssf.usermodel.HSSFCell;
import org.apache.poi.hssf.usermodel.HSSFRow;
import org.apache.poi.hssf.usermodel.HSSFSheet;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

class DynamicTemplate_height_0_3_Test {

    @ExtendWith(MockitoExtension.class)
    public class TestDynamicTemplate_height_0_3 {

        @Test
        void testHeight() {
            // Arrange
            String name = "Example Template";
            HSSFSheet sheet = mock(HSSFSheet.class);
            int height = 10;
            List<NamedStyle> styles = mock(List.class);
            // Act
            DynamicTemplate dynamicTemplate = new DynamicTemplate(name, sheet, height, styles);
            int actualHeight = dynamicTemplate.height();
            // Assert
            assertEquals(height, actualHeight);
        }
    }
}
