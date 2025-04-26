package org.apache.poi.hssf.usermodel;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.junit.MockitoJUnitRunner;
import java.util.List;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.apache.poi.hssf.model.Workbook;
import org.apache.poi.hssf.record.FormatRecord;
import java.util.Iterator;
import java.util.ListIterator;
import java.util.Vector;

@RunWith(MockitoJUnitRunner.class)
public class HSSFDataFormat_getFormat_3_4_Test {

    @Mock
    private Workbook workbook;

    @InjectMocks
    private HSSFDataFormat focal;

    @Test
    public void testGetFormat_BuiltinFormatFound() {
        // Arrange
        String format = "TEXT";
        // Act
        short formatIndex = focal.getFormat(format);
        // Assert
        assertEquals(-1, formatIndex);
    }

    @Test
    public void testGetFormat_BuiltinFormatNotFound() {
        // Arrange
        String format = "CUSTOM_FORMAT";
        // Act
        short formatIndex = focal.getFormat(format);
        // Assert
        assertEquals(0, formatIndex);
    }

    @Test
    public void testGetFormat_MovedBuiltins() {
        // Arrange
        String format = "TEXT";
        // Act
        short formatIndex = focal.getFormat(format);
        // Assert
        assertEquals(0, formatIndex);
    }
}
