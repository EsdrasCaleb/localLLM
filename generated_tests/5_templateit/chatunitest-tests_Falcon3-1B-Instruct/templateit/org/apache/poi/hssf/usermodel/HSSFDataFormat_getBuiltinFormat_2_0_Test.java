// Fix the Buggy Line:
package org.apache.poi.hssf.usermodel;

import org.junit.jupiter.api.extension.ExtendWith;
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

class HSSFDataFormat_getBuiltinFormat_2_0_Test {

    @Test
    public void testGetBuiltinFormat() {
        // Test cases for both positive and negative inputs
        assertEquals(0, HSSFDataFormat.getBuiltinFormat("@"), "Expected 0");
        assertEquals(1, HSSFDataFormat.getBuiltinFormat("@TEXT"), "Expected 1");
        assertEquals(2, HSSFDataFormat.getBuiltinFormat("TEXT"), "Expected 2");
        assertEquals(3, HSSFDataFormat.getBuiltinFormat("TEXT@"), "Expected 3");
        assertEquals(4, HSSFDataFormat.getBuiltinFormat("@TEXT@"), "Expected 4");
        // Test with invalid format
        assertEquals(-1, HSSFDataFormat.getBuiltinFormat("abc"), "Expected -1");
        // Test with non-existent format
        assertEquals(-1, HSSFDataFormat.getBuiltinFormat("abc@"), "Expected -1");
    }
}
