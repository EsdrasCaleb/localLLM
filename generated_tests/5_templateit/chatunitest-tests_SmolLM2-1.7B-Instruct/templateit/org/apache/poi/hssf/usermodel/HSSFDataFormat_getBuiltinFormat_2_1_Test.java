package org.apache.poi.hssf.usermodel;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.apache.poi.hssf.model.Workbook;
import org.apache.poi.hssf.record.FormatRecord;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;
import java.util.Vector;

@ExtendWith(MockitoExtension.class)
public class HSSFDataFormat_getBuiltinFormat_2_1_Test {

    @Mock
    private Workbook workbook;

    @InjectMocks
    private HSSFDataFormat focal;

    @Test
    void testGetBuiltinFormat() {
        // Arrange
        String format = "TEXT";
        // Act
        short formatCode = focal.getBuiltinFormat(format);
        // Assert
        assertEquals(HSSFDataFormat.getBuiltinFormat(format), formatCode);
    }
}
