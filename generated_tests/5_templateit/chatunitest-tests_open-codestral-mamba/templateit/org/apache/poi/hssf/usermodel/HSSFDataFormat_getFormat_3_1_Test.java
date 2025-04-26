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

@ExtendWith(MockitoExtension.class)
public class HSSFDataFormat_getFormat_3_1_Test {

    @Mock
    private Workbook workbook;

    @InjectMocks
    private HSSFDataFormat hssfDataFormat;

    @BeforeEach
    public void setUp() {
        when(workbook.getFormat("TEXT", true)).thenReturn(Short.valueOf("1"));
    }

    @Test
    public void testGetFormat_ExistingFormat() {
        short formatIndex = hssfDataFormat.getFormat("TEXT");
        assertEquals(1, formatIndex);
    }

    @Test
    public void testGetFormat_NewFormat() {
        when(workbook.getFormat("NEW_FORMAT", true)).thenReturn(Short.valueOf("-1"));
        short formatIndex = hssfDataFormat.getFormat("NEW_FORMAT");
        assertEquals(-1, formatIndex);
    }
}
