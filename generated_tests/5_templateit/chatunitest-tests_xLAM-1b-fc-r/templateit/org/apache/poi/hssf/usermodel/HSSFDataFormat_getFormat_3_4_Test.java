package org.apache.poi.hssf.usermodel;

import org.mockito.junit.jupiter.MockitoExtension;
import org.apache.poi.hssf.model.Workbook;
import org.apache.poi.hssf.record.FormatRecord;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;
import java.util.Vector;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;

public class HSSFDataFormat_getFormat_3_4_Test {

    @Mock
    HSSFDataFormat hssfDataFormat;

    @BeforeEach
    public void setup() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void getFormat_returnsCorrectIndex_whenFormatExists() {
        // Given
        String format = "TEXT";
        short expectedIndex = 0;
        when(hssfDataFormat.getFormat(format)).thenReturn(expectedIndex);
        // When
        short actualIndex = hssfDataFormat.getFormat(format);
        // Then
        assertEquals(expectedIndex, actualIndex);
    }

    @Test
    public void getFormat_returnsCorrectIndex_whenFormatDoesNotExist() {
        // Given
        String format = "UNKNOWN";
        short expectedIndex = -1;
        when(hssfDataFormat.getFormat(format)).thenReturn(expectedIndex);
        // When
        short actualIndex = hssfDataFormat.getFormat(format);
        // Then
        assertEquals(expectedIndex, actualIndex);
    }

    @Test
    public void getFormat_returnsCorrectIndex_whenFormatIsBuiltin() {
        // Given
        String format = "@";
        short expectedIndex = 0;
        when(hssfDataFormat.getFormat(format)).thenReturn(expectedIndex);
        // When
        short actualIndex = hssfDataFormat.getFormat(format);
        // Then
        assertEquals(expectedIndex, actualIndex);
    }
}
