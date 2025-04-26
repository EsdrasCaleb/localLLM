package org.apache.poi.hssf.usermodel;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.apache.poi.hssf.model.Workbook;
import org.apache.poi.hssf.record.FormatRecord;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;
import java.util.Vector;

public class HSSFDataFormat_getBuiltinFormat_2_1_Test {

    @Test
    public void testGetBuiltinFormat() {
        // Test with valid input
        assertEquals(Short.valueOf("0"), HSSFDataFormat.getBuiltinFormat("TEXT"));
        // Test with invalid input
        Throwable exception = assertThrows(IllegalArgumentException.class, () -> {
            HSSFDataFormat.getBuiltinFormat("INVALID");
        });
        assertEquals("Invalid format: INVALID", exception.getMessage());
    }
}
