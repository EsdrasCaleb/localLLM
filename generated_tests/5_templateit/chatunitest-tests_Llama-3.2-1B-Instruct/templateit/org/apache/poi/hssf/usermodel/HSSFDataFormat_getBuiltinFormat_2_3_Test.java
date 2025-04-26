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
public class HSSFDataFormat_getBuiltinFormat_2_3_Test {

    @Mock
    private HSSFDataFormat focal;

    @InjectMocks
    private HSSFDataFormat instance;

    @Test
    public void testGetBuiltinFormat() {
        // Arrange
        when(focal.getBuiltinFormats()).thenReturn(new Vector());
        // Act
        short result = instance.getBuiltinFormat("TEXT");
        // Assert
        assertEquals(0, result);
    }
}
