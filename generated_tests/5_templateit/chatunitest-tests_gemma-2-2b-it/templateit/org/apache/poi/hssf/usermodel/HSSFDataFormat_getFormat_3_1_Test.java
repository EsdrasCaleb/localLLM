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

class HSSFDataFormat_getFormat_3_1_Test {

    @Test
    void testGetFormat() {
        HSSFDataFormat hssfDataFormat = new HSSFDataFormat(new Workbook());
        assertEquals(0, hssfDataFormat.getFormat("TEXT"));
        assertEquals(0, hssfDataFormat.getFormat("TEXT"));
    }
}
